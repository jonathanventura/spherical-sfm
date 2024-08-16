#pragma once

#include <vector>
#include <string>
#include <map>

#include <Eigen/Core>

namespace COLMAP {

    struct Keypoint
    {
        Eigen::Vector2d x;
        uint8_t desc[128];
    };

    struct Image
    {
        std::string name;
        int width, height;
        double f, cx, cy;
        Eigen::Matrix3d get_K() const { Eigen::Matrix3d K = Eigen::Matrix3d::Identity(); K(0,0) = K(1,1) = f; K(0,2) = cx; K(1,2) = cy; return K; }
        std::vector<Keypoint> keypoints;
    };

    struct Match
    {
        uint32_t image_id1, image_id2;
        std::vector< std::pair<uint32_t,uint32_t> > matches;
    };

    struct Database
    {
        std::string dbpath;
        Database( const std::string &_dbpath ) : dbpath(_dbpath) { }
    
        bool read();
        bool write();

        uint32_t find( const std::string &name ) const;

        std::map<uint32_t,Image> images;
        std::map<std::string,uint32_t> image_ids;
        std::vector<Match> matches;
    };

}


