#include <iostream>
#include <sqlite3.h>

#include "colmap.h"

const char *CREATE_TABLES = 
"CREATE TABLE cameras (camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,"
"                      model INTEGER NOT NULL,"
"                      width INTEGER NOT NULL,"
"                      height INTEGER NOT NULL,"
"                      params BLOB,"
"                      prior_focal_length INTEGER NOT NULL);"
"CREATE TABLE images (image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,"
"                     name TEXT NOT NULL UNIQUE,"
"                     camera_id INTEGER NOT NULL,"
"                     prior_qw REAL,"
"                     prior_qx REAL,"
"                     prior_qy REAL,"
"                     prior_qz REAL,"
"                     prior_tx REAL,"
"                     prior_ty REAL,"
"                     prior_tz REAL,"
"                     CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < 2147483647),"
"                     FOREIGN KEY(camera_id) REFERENCES cameras(camera_id));"
"CREATE UNIQUE INDEX index_name ON images(name);"
"CREATE TABLE keypoints (image_id INTEGER PRIMARY KEY NOT NULL,"
"                        rows INTEGER NOT NULL,"
"                        cols INTEGER NOT NULL,"
"                        data BLOB,"
"                        FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE);"
"CREATE TABLE descriptors (image_id  INTEGER PRIMARY KEY NOT NULL,"
"                        rows INTEGER NOT NULL,"
"                        cols INTEGER NOT NULL,"
"                        data BLOB,"
"                        FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE);"
"CREATE TABLE matches (pair_id INTEGER PRIMARY KEY NOT NULL,"
"                        rows INTEGER NOT NULL,"
"                        cols INTEGER NOT NULL,"
"                        data BLOB);"
"CREATE TABLE two_view_geometries (pair_id INTEGER PRIMARY KEY NOT NULL,"
"                        rows INTEGER NOT NULL,"
"                        cols INTEGER NOT NULL,"
"                        data BLOB,"
"                        config INTEGER NOT NULL,"
"                        F BLOB,"
"                        H BLOB,"
"                        qvec BLOB,"
"                        tvec BLOB)"
;

namespace COLMAP {

    static void pair_id_to_image_ids(uint64_t pair_id, uint32_t &image_id1, uint32_t &image_id2 )
    {
        image_id2 = pair_id % 2147483647;
        image_id1 = (pair_id - image_id2) / 2147483647;
    }

    static bool open_connection(const char *path, sqlite3 **conn)
    {
        if ( sqlite3_open( path, conn ) != SQLITE_OK )
        {
            std::cerr << "could not open " << path << "\n";
            std::cerr << "sqlite3 error: " << sqlite3_errmsg(*conn) << "\n";
            return false;
        }
        return true;
    }

    static bool begin_transaction(sqlite3 *conn)
    {
        if ( sqlite3_exec(conn,"BEGIN TRANSACTION",NULL,NULL,NULL) != SQLITE_OK )
        {
            std::cerr << "error on begin transaction: " << sqlite3_errmsg(conn) << "\n";
            return false;
        }
        return true;
    }

    static bool end_transaction(sqlite3 *conn)
    {
        if ( sqlite3_exec(conn,"END TRANSACTION",NULL,NULL,NULL) != SQLITE_OK )
        {
            std::cerr << "error on end transaction: " << sqlite3_errmsg(conn) << "\n";
            return false;
        }
        return true;
    }

    static bool execute_sql(sqlite3 *conn, const char *sql)
    {
        if ( sqlite3_exec(conn,sql,NULL,NULL,NULL) != SQLITE_OK )
        {
            std::cerr << "error executing sql: " << sqlite3_errmsg(conn) << "\n\n";
            return false;
        }
        return true;
    }

    bool Database::read()
    {
        sqlite3 *conn;
        if ( !open_connection( dbpath.c_str(), &conn ) ) return false;

        sqlite3_stmt *image_stmt;
        sqlite3_prepare_v2(conn,"SELECT images.image_id, images.name, cameras.model, cameras.width, cameras.height, cameras.params FROM images JOIN cameras ON images.camera_id=cameras.camera_id",-1,&image_stmt,NULL);
        while ( sqlite3_step(image_stmt) == SQLITE_ROW )
        {
            uint32_t image_id = sqlite3_column_int(image_stmt,0);
            Image image;
            image.name = std::string(reinterpret_cast<const char*>(sqlite3_column_text(image_stmt, 1)));
            if ( sqlite3_column_int(image_stmt,2) !=0 )
            {
                std::cerr << "error: camera model is not SIMPLE_PINHOLE\n";
                sqlite3_close( conn );
                return false;
            }
            image.width = sqlite3_column_int(image_stmt,3);
            image.height = sqlite3_column_int(image_stmt,4);
            const double *params = (const double*)sqlite3_column_blob(image_stmt, 5);
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

        sqlite3_stmt *descriptors_stmt;
        sqlite3_prepare_v2(conn,"SELECT image_id, rows, cols, data FROM descriptors",-1,&descriptors_stmt,NULL);
        while ( sqlite3_step(descriptors_stmt) == SQLITE_ROW )
        {
            uint32_t image_id = sqlite3_column_int(descriptors_stmt,0);
            int rows = sqlite3_column_int(descriptors_stmt,1);
            int cols = sqlite3_column_int(descriptors_stmt,2);
            uint8_t *data = (uint8_t *)sqlite3_column_blob(descriptors_stmt,3);
            uint8_t *dataptr = data;
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

        return true;
    }

    bool Database::write()
    {
        sqlite3 *conn;
        if ( !open_connection( dbpath.c_str(), &conn ) ) return false;

        if ( !execute_sql(conn, CREATE_TABLES ) ) { sqlite3_close(conn); return false; }

        if ( images.size() == 0 ) 
        {
            std::cerr << "error: no images\n";
            sqlite3_close( conn );
            return false;
        }
        
        if ( !begin_transaction(conn) ) { sqlite3_close(conn); return false; }

        sqlite3_stmt *insert_camera_stmt;
        sqlite3_prepare_v2(conn,"INSERT INTO cameras (camera_id,model,width,height,params,prior_focal_length) values (?,?,?,?,?,?);",-1,&insert_camera_stmt,NULL);
        for ( auto im : images )
        {
            double data[3] = { im.second.f, im.second.cx, im.second.cy };
            sqlite3_bind_int64(insert_camera_stmt, 1, 1);
            sqlite3_bind_int64(insert_camera_stmt, 2, 0);
            sqlite3_bind_int64(insert_camera_stmt, 3, im.second.width);
            sqlite3_bind_int64(insert_camera_stmt, 4, im.second.height);
            sqlite3_bind_blob(insert_camera_stmt, 5, data, sizeof(double)*3, SQLITE_TRANSIENT );
            sqlite3_bind_int64(insert_camera_stmt, 6, 0);
            if ( sqlite3_step(insert_camera_stmt) != SQLITE_DONE )
            {
                std::cerr << "could not insert camera: " << sqlite3_errmsg(conn) << "\n";
                sqlite3_close( conn );
                return false;
            }
            sqlite3_clear_bindings( insert_camera_stmt );
            sqlite3_reset( insert_camera_stmt );
            
            break;
        }
        if ( !end_transaction(conn) ) { sqlite3_close(conn); return false; }
        sqlite3_finalize(insert_camera_stmt);

        if ( !begin_transaction(conn) ) { sqlite3_close(conn); return false; }

        sqlite3_stmt *insert_image_stmt;
        sqlite3_prepare_v2(conn,"INSERT INTO images (name,camera_id) values (?,?);",-1,&insert_image_stmt,NULL);
        for ( auto im : images )
        {
            sqlite3_bind_text(insert_image_stmt, 1, im.second.name.c_str(), im.second.name.length(), SQLITE_STATIC);
            sqlite3_bind_int64(insert_image_stmt, 2, 1);
            if ( sqlite3_step(insert_image_stmt) != SQLITE_DONE )
            {
                std::cerr << "could not insert image: " << sqlite3_errmsg(conn) << "\n";
                sqlite3_close( conn );
                return false;
            }
            sqlite3_clear_bindings( insert_image_stmt );
            sqlite3_reset( insert_image_stmt );
        }
        if ( !end_transaction(conn) ) { sqlite3_close(conn); return false; }
        sqlite3_finalize(insert_image_stmt);

        if ( !begin_transaction(conn) ) { sqlite3_close(conn); return false; }
        sqlite3_stmt *insert_keypoints_stmt;
        sqlite3_stmt *insert_descriptors_stmt;
        sqlite3_prepare_v2(conn,"INSERT INTO keypoints (image_id,rows,cols,data) values (?,?,?,?);",-1,&insert_keypoints_stmt,NULL);
        sqlite3_prepare_v2(conn,"INSERT INTO descriptors (image_id,rows,cols,data) values (?,?,?,?);",-1,&insert_descriptors_stmt,NULL);
        for ( auto im : images )
        {
            float *kpts_data = new float[im.second.keypoints.size() * 6];
            uint8_t *desc_data = new uint8_t[im.second.keypoints.size() * 128];
            float *kpts_data_ptr = kpts_data;
            uint8_t *desc_data_ptr = desc_data;
            for ( auto kp : im.second.keypoints )
            {
                kpts_data_ptr[0] = kp.x(0);
                kpts_data_ptr[1] = kp.x(1);
                kpts_data_ptr[2] = 1;
                kpts_data_ptr[3] = 0;
                kpts_data_ptr[4] = 0;
                kpts_data_ptr[5] = 1;
                std::memcpy(desc_data_ptr,kp.desc,128);
                kpts_data_ptr += 6;
                desc_data_ptr += 128;
            }
            sqlite3_bind_int(insert_keypoints_stmt, 1, im.first);
            sqlite3_bind_int(insert_keypoints_stmt, 2, im.second.keypoints.size());
            sqlite3_bind_int(insert_keypoints_stmt, 3, 6);
            sqlite3_bind_blob(insert_keypoints_stmt, 4, kpts_data, sizeof(float)*6*im.second.keypoints.size(), SQLITE_TRANSIENT);

            if ( sqlite3_step(insert_keypoints_stmt) != SQLITE_DONE )
            {
                std::cerr << "could not insert keypoints: " << sqlite3_errmsg(conn) << "\n";
                sqlite3_close( conn );
                return false;
            }

            delete [] kpts_data;
            delete [] desc_data;

            sqlite3_reset(insert_keypoints_stmt);
            sqlite3_reset(insert_descriptors_stmt);
            sqlite3_clear_bindings(insert_keypoints_stmt);
            sqlite3_clear_bindings(insert_descriptors_stmt);
        }
        if ( !end_transaction(conn) ) { sqlite3_close(conn); return false; }
        sqlite3_finalize(insert_keypoints_stmt);
        sqlite3_finalize(insert_descriptors_stmt);

        sqlite3_close( conn );
        return true;
    }

    uint32_t Database::find( const std::string &name ) const 
    {
        auto it = image_ids.find(name);
        if ( it == image_ids.end() ) return -1;
        return it->second;
    }

}

