#pragma once

namespace sphericalsfm {
    // from http://stackoverflow.com/questions/311703/algorithm-for-sampling-without-replacement
    // Algorithm 3.4.2S from D. Knuth, "Seminumeric Algorithms."
    template<typename T>
    void random_sample( T begin, int N, T sample_begin, int n )
    {
        int t = 0; // total input records dealt with
        int m = 0; // number of items selected so far
        double u;
        
        while (m < n)
        {
            u = rand()/(double)RAND_MAX; // call a uniform(0,1) random number generator
            
            if ( (N - t)*u >= n - m )
            {
                t++;
            }
            else
            {
                *(sample_begin+m) = *(begin+t);
                t++; m++;
            }
        }
    }

    template<typename ListType, typename EstimatorType>
    struct MSAC {
        double inlier_threshold;
        double prob_success;
        double init_outlier_ratio;
        int iter; // number of iterations performed
        
        MSAC( double _prob_success = 0.999, double _init_outlier_ratio = 0.8 )

        : prob_success( _prob_success ), init_outlier_ratio( _init_outlier_ratio )
        {
            
        }

        void compute_num_iterations( double outlier_ratio, int sample_size, double &num_iter )
        {
            double num = log(1.-prob_success);
            double den = log(1.-pow(1.-outlier_ratio,sample_size));
            num_iter = num/den;
        }
        
        void count_inliers( EstimatorType *estimator, double threshsq,
                           typename ListType::iterator begin, typename ListType::iterator end, std::vector<bool> &my_inliers,
                           int &my_num_inliers, double &my_score )
        {
            my_score = 0;
            my_num_inliers = 0;
            int j = 0;
            for ( typename ListType::iterator it = begin; it != end; it++, j++ )
            {
                double score = estimator->score(it);
                my_inliers[j] = ( score <= threshsq );
                if ( my_inliers[j] ) {
                    my_score += score;
                    my_num_inliers++;
                } else {
                    my_score += threshsq;
                }
            }
        }
        
        int compute( typename ListType::iterator begin, typename ListType::iterator end, std::vector<EstimatorType*> &estimators, EstimatorType **best_estimator, std::vector<bool> &inliers )
        {
            double threshsq = inlier_threshold*inlier_threshold;
            size_t N = std::distance(begin,end);
            int m = estimators[0]->sampleSize();
            double outlier_ratio = init_outlier_ratio;
            
            ListType random_subset( m );
            
            double num_iter = estimators.size();
            
            double best_score = INFINITY;
            int num_inliers = 0;
            std::vector<bool> my_inliers(N);
            for ( int i = 0; i < N; i++ ) my_inliers[i] = false;
            
            iter = 0;
            while ( iter < num_iter && iter < estimators.size() )
            {
                // generate a random sample
                random_sample(begin, N, random_subset.begin(), m );
                
                // compute solution
                int nsolns = estimators[iter]->compute( random_subset.begin(), random_subset.end() );
                
                int best_index = -1;
                for ( int i = 0; i < nsolns; i++ )
                {
                    estimators[iter]->chooseSolution(i);
                    
                    double my_score = 0;
                    int my_num_inliers = 0;
                    count_inliers( estimators[iter], threshsq, begin, end, my_inliers, my_num_inliers, my_score );
                    
                    // save solution if better
                    if ( my_score < best_score )
                    {
                        best_score = my_score;
                        num_inliers = my_num_inliers;
                        best_index = i;
                        inliers = my_inliers;
                    }
                }
                if ( best_index >= 0 )
                {
                    estimators[iter]->chooseSolution(best_index);
                    *best_estimator = estimators[iter];
                }

                // re-compute number of iterations
                outlier_ratio = (N-num_inliers)/(double)N;
                if ( outlier_ratio < 1.0 )
                {
                    compute_num_iterations(outlier_ratio,m,num_iter);
    //                std::cout << "num iterations now: " << num_iter << "\n";
                    if ( num_iter > estimators.size() ) num_iter = estimators.size();
                }
                
                iter++;
            }
            
            return num_inliers;
        }
    };
}

