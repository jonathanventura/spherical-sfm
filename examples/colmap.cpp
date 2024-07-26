#include <iostream>
#include <sqlite3.h>

#include "colmap.h"


namespace COLMAP {

    static void pair_id_to_image_ids(uint64_t pair_id, uint32_t &image_id1, uint32_t &image_id2 )
    {
        image_id2 = pair_id % 2147483647;
        image_id1 = (pair_id - image_id2) / 2147483647;
    }

    Database::Database( const std::string &dbpath )
    {
        sqlite3 *conn;
        sqlite3_open( dbpath.c_str(), &conn );

        sqlite3_stmt *image_stmt;
        sqlite3_prepare_v2(conn,"SELECT images.image_id, images.name, cameras.params FROM images JOIN cameras ON images.camera_id=cameras.camera_id",-1,&image_stmt,NULL);
        while ( sqlite3_step(image_stmt) == SQLITE_ROW )
        {
            uint32_t image_id = sqlite3_column_int(image_stmt,0);
            Image image;
            image.name = std::string(reinterpret_cast<const char*>(sqlite3_column_text(image_stmt, 1)));
            const double *params = (const double*)sqlite3_column_blob(image_stmt, 2);
            image.f = params[0];
            image.cx = params[1];
            image.cy = params[2];
            images[image_id] = image;
            image_ids[image.name] = image_id;
        }

        sqlite3_stmt *keypoints_stmt;
        sqlite3_prepare_v2(conn,"SELECT image_id, rows, cols, data FROM keypoints",-1,&keypoints_stmt,NULL);
        while ( sqlite3_step(keypoints_stmt) == SQLITE_ROW )
        {
            uint32_t image_id = sqlite3_column_int(keypoints_stmt,0);
            int rows = sqlite3_column_int(keypoints_stmt,1);
            int cols = sqlite3_column_int(keypoints_stmt,2);
            float *data = (float *)sqlite3_column_blob(keypoints_stmt,3);
            float *dataptr = data;
            for ( int i = 0; i < rows; i++,dataptr += cols )
            {
                Keypoint keypoint;
                keypoint.x(0) = dataptr[0];
                keypoint.x(1) = dataptr[1];
                images[image_id].keypoints.push_back(keypoint);
            } 
        }

        sqlite3_stmt *match_stmt;
        sqlite3_prepare_v2(conn,"SELECT pair_id,rows,cols,data FROM matches",-1,&match_stmt,NULL);
        while ( sqlite3_step(match_stmt) == SQLITE_ROW )
        {
            uint64_t pair_id = sqlite3_column_int64(match_stmt,0);
            uint32_t image_id1, image_id2;
            pair_id_to_image_ids(pair_id,image_id1,image_id2);
            Match match;
            match.image_id1 = image_id1;
            match.image_id2 = image_id2;
            int rows = sqlite3_column_int(match_stmt,1);
            int cols = sqlite3_column_int(match_stmt,2);
            uint32_t *data = (uint32_t *)sqlite3_column_blob(match_stmt,3);
            uint32_t *dataptr = data;
            for ( int i = 0; i < rows; i++,dataptr += 2 )
            {
                match.matches.push_back(std::make_pair(dataptr[0],dataptr[1]));
            } 
            matches.push_back(match);
        }

        sqlite3_close( conn );
    }

    uint32_t Database::find( const std::string &name ) const 
    {
        auto it = image_ids.find(name);
        if ( it == image_ids.end() ) return -1;
        return it->second;
    }

}

