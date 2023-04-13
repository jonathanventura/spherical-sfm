#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include <RansacLib/sampling.h>
#include <RansacLib/utils.h>
#include <RansacLib/ransac.h>

namespace ransac_lib {

template <class Model, class ModelVector, class Solver,
          class Sampler = UniformSampling<Solver> >
class VanillaMSAC : public RansacBase {
 public:
  // Estimates a model using a given solver. 
  // Returns the number of inliers.
  int EstimateModel(const RansacOptions& options, const Solver& solver,
                    Model* best_model, RansacStatistics* statistics) const {
    ResetStatistics(statistics);
    RansacStatistics& stats = *statistics;

    // Sanity check: No need to run RANSAC if there are not enough data
    // points.
    const int kMinSampleSize = solver.min_sample_size();
    const int kNumData = solver.num_data();
    if (kMinSampleSize > kNumData || kMinSampleSize <= 0) {
      return 0;
    }

    // Initializes variables, etc.
    Sampler sampler(options.random_seed_, solver);
    std::mt19937 rng;
    rng.seed(options.random_seed_);

    uint32_t max_num_iterations =
        std::max(options.max_num_iterations_, options.min_num_iterations_);

    const double kSqrInlierThresh = options.squared_inlier_threshold_;

    Model best_minimal_model;
    double best_min_model_score = std::numeric_limits<double>::max();

    std::vector<int> minimal_sample(kMinSampleSize);
    ModelVector estimated_models;

    // Runs random sampling.
    for (stats.num_iterations = 0u; stats.num_iterations < max_num_iterations;
         ++stats.num_iterations) {
      sampler.Sample(&minimal_sample);

      // MinimalSolver returns the number of estimated models.
      const int kNumEstimatedModels =
          solver.MinimalSolver(minimal_sample, &estimated_models);
      if (kNumEstimatedModels <= 0) continue;

      // Finds the best model among all estimated models.
      double best_local_score = std::numeric_limits<double>::max();
      int best_local_model_id = 0;
      GetBestEstimatedModelId(solver, estimated_models, kNumEstimatedModels,
                              kSqrInlierThresh, &best_local_score,
                              &best_local_model_id);

      // Updates the best model found so far.
      if (best_local_score < best_min_model_score ) {
        const bool kBestMinModel = best_local_score < best_min_model_score;

        if (kBestMinModel) {
          // New best model (estimated from inliers found. Stores this model
          // and runs local optimization.
          best_min_model_score = best_local_score;
          best_minimal_model = estimated_models[best_local_model_id];

          // Updates the best model.
          UpdateBestModel(best_min_model_score, best_minimal_model,
                          &(stats.best_model_score), best_model);
        }

        if (!kBestMinModel) continue;

        // Updates the number of RANSAC iterations.
        stats.best_num_inliers = GetInliers(
            solver, *best_model, kSqrInlierThresh, &(stats.inlier_indices));
        stats.inlier_ratio = static_cast<double>(stats.best_num_inliers) /
                             static_cast<double>(kNumData);
        max_num_iterations = utils::NumRequiredIterations(
            stats.inlier_ratio, 1.0 - options.success_probability_,
            kMinSampleSize, options.min_num_iterations_,
            options.max_num_iterations_);
      }
    }

    return stats.best_num_inliers;
  }

 protected:
  void GetBestEstimatedModelId(const Solver& solver, const ModelVector& models,
                               const int num_models,
                               const double squared_inlier_threshold,
                               double* best_score, int* best_model_id) const {
    *best_score = std::numeric_limits<double>::max();
    *best_model_id = 0;
    for (int m = 0; m < num_models; ++m) {
      double score = std::numeric_limits<double>::max();
      ScoreModel(solver, models[m], squared_inlier_threshold, &score);

      if (score < *best_score) {
        *best_score = score;
        *best_model_id = m;
      }
    }
  }

  void ScoreModel(const Solver& solver, const Model& model,
                  const double squared_inlier_threshold, double* score) const {
    const int kNumData = solver.num_data();
    *score = 0.0;
    for (int i = 0; i < kNumData; ++i) {
      double squared_error = solver.EvaluateModelOnPoint(model, i);
      *score += ComputeScore(squared_error, squared_inlier_threshold);
    }
  }

  // MSAC (top-hat) scoring function.
  inline double ComputeScore(const double squared_error,
                             const double squared_error_threshold) const {
    return std::min(squared_error, squared_error_threshold);
  }

  int GetInliers(const Solver& solver, const Model& model,
                 const double squared_inlier_threshold,
                 std::vector<int>* inliers) const {
    const int kNumData = solver.num_data();
    if (inliers == nullptr) {
      int num_inliers = 0;
      for (int i = 0; i < kNumData; ++i) {
        double squared_error = solver.EvaluateModelOnPoint(model, i);
        if (squared_error < squared_inlier_threshold) {
          ++num_inliers;
        }
      }
      return num_inliers;
    } else {
      inliers->clear();
      int num_inliers = 0;
      for (int i = 0; i < kNumData; ++i) {
        double squared_error = solver.EvaluateModelOnPoint(model, i);
        if (squared_error < squared_inlier_threshold) {
          ++num_inliers;
          inliers->push_back(i);
        }
      }
      return num_inliers;
    }
  }

  inline void UpdateBestModel(const double score_curr, const Model& m_curr,
                              double* score_best, Model* m_best) const {
    if (score_curr < *score_best) {
      *score_best = score_curr;
      *m_best = m_curr;
    }
  }
};

}  // namespace ransac_lib
