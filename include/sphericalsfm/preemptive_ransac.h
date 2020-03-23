
#pragma once

#include <iostream>

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
    struct PreemptiveRANSAC {
        double inlier_threshold;
        size_t B;
        
        /**
         * \param _B Block size
         */
        PreemptiveRANSAC( size_t _B = 10 )
        : inlier_threshold( 0.001 ), B( _B )
        {
            
        }
        
        int compute( typename ListType::iterator begin, typename ListType::iterator end, std::vector<EstimatorType*> &estimators, EstimatorType **best_estimator, std::vector<bool> &inliers )
        {
            double threshsq = inlier_threshold*inlier_threshold;
            size_t M = estimators.size();
            size_t N = std::distance(begin,end);
            int m = estimators[0]->sampleSize();
            
            ListType random_subset( m+1 );
            
            // this contains the number of inliers (first) for a hypothesis index (second)
            std::vector< std::pair<int,size_t> > numInliersAndIndex(estimators.size());
            
            // generate M hypotheses
            for ( size_t i = 0; i < M; i++ )
            {
                numInliersAndIndex[i].first = 0;
                numInliersAndIndex[i].second = i;
                
                // generate a random sample with one extra correspondence for disambiguation
                random_sample(begin, N, random_subset.begin(), m+1 );
                
                // compute solution
                int nsolns = estimators[i]->compute( random_subset.begin(), random_subset.begin()+m );
                
                // if bad sample, continue
                if ( nsolns == 0 ) continue;
                
                // if more than one solution, disambiguate
                if ( nsolns > 1 )
                {
                    // choose solution which has best score for extra correspondence
                    int best_index = 0;
                    double best_score = INFINITY;
                    for ( int j = 0; j < nsolns; j++ )
                    {
                        estimators[i]->chooseSolution(j);
                        double score = estimators[i]->score( random_subset.begin()+m );
                        if ( score < best_score )
                        {
                            best_score = score;
                            best_index = j;
                        }
                    }
                    estimators[i]->chooseSolution(best_index);
                }
            }
            
            // test blocks of correspondences
            typename ListType::iterator it = begin;
            size_t f_i = M;
            for ( size_t i = 1; i < N; i++ )
            {
                // increment number of inliers for all remaining hypotheses using this block of correspondences
                typename ListType::iterator start = it;
                for ( ; it != start+B && it != end; ++it )
                {
                    for ( size_t j = 0; j < f_i; j++ )
                    {
                        double score = estimators[ numInliersAndIndex[j].second ]->score(it);
                        if ( score <= threshsq ) numInliersAndIndex[j].first++;
                    }
                }
                
                // f_i is the number of remaining hypotheses
                f_i = floor(M*pow(2.,-floor(i/B)));
                
                // partially sort remaining hypotheses by num inliers
                std::partial_sort( numInliersAndIndex.begin(), numInliersAndIndex.begin()+f_i, numInliersAndIndex.end(), std::greater< std::pair<int,size_t> >() );
                
                // stop if we have one hypothesis left
                if ( f_i <= 1 ) break;
                
                // stop if we run out of correspondences
                if ( it == end ) break;
            }
            
            // top hypothesis is at beginning of list
            (*best_estimator) = estimators[numInliersAndIndex[0].second];
            
            // evaluate all correspondences to determine inliers
            int num_inliers = 0;
            inliers.resize( N );
            
            for ( size_t i = 0; i < N; i++ )
            {
                inliers[i] = false;
                double score = (*best_estimator)->score(begin+i);
                if ( score <= threshsq )
                {
                    inliers[i] = true;
                    num_inliers++;
                }
            }
            
            return num_inliers;
        }
    };
}

