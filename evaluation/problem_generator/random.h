#pragma once

#include <random>
#include <Eigen/Core>

namespace problem_generator
{
    static std::default_random_engine engine;

    class RandomGenerator
    {
        std::uniform_real_distribution<double> uniform;
        std::normal_distribution<double> normal;
    public:
        RandomGenerator() : uniform(-1,1) { }
        
        double rand() { return uniform(engine); }
        double randn() { return normal(engine); }
        Eigen::Vector2d rand2() { return Eigen::Vector2d(rand(),rand()); }
        Eigen::Vector3d rand3() { return Eigen::Vector3d(rand(),rand(),rand()); }

        Eigen::Vector2d randn2() { return Eigen::Vector2d(randn(),randn()); }
        Eigen::Vector3d randn3() { return Eigen::Vector3d(randn(),randn(),randn()); }

        Eigen::Vector3d rand_unit_vector() { const Eigen::Vector3d v = randn3(); return v/v.norm(); }
    };
/*
    // sample random 3-vector from standard normal distribution
    Eigen::Vector3d randn_double3() { Eigen::Vector3d v; v << randn_double(),randn_double(),randn_double(); return v; }

    // sample random point on unit sphere
    Eigen::Vector3d rand_unit_vector() { Eigen::Vector3d v; v = randn_double3(); v/=v.norm(); return v; }
    }
    // sample random double [-1 1] range
    // double rand_double() { return ((double)rand()/RAND_MAX)*2.-1.; }

    // sample random 2-vector on [-1 1] range
    // Eigen::Vector2d rand_double2() { Eigen::Vector2d v; v << rand_double(),rand_double(); return v; }

    // sample random 3-vector on [-1 1] range
    // Eigen::Vector3d rand_double3() { Eigen::Vector3d v; v << rand_double(),rand_double(),rand_double(); return v; }

    // sample random double from standard normal distribution
    double randn_double() {

        double r,v1,v2,fac;

        r=2;
        while (r>=1) {
            v1=(2*((double)rand()/(double)RAND_MAX)-1);
            v2=(2*((double)rand()/(double)RAND_MAX)-1);
            r=v1*v1+v2*v2;
        }
        fac=sqrt(-2*log(r)/r);

        return(v2*fac);

    }

    // sample random 2-vector from standard normal distribution
    Eigen::Vector2d randn_double2() { Eigen::Vector2d v; v << randn_double(),randn_double(); return v; }

    // sample random 3-vector from standard normal distribution
    Eigen::Vector3d randn_double3() { Eigen::Vector3d v; v << randn_double(),randn_double(),randn_double(); return v; }

    // sample random point on unit sphere
    Eigen::Vector3d rand_unit_vector() { Eigen::Vector3d v; v = randn_double3(); v/=v.norm(); return v; }
*/
}
