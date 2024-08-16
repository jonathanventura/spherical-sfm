
#include <stdio.h> 

#include "spherical_sfm_io.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

namespace sphericalsfmtools {

    void write_feature_tracks( const std::string &outputpath, const std::vector<Keyframe> &keyframes, const std::vector<ImageMatch> &image_matches )
    {
        std::string keyframespath = outputpath + "/keyframes.txt";
        FILE *keyframesf = fopen( keyframespath.c_str(), "w" );
        fprintf(keyframesf,"%d\n",keyframes.size());
        for ( int i = 0; i < keyframes.size(); i++ )
        {
            fprintf(keyframesf,"%d %s\n",keyframes[i].index,keyframes[i].name);
        }
        fclose(keyframesf);

        FILE *featuresf = fopen((outputpath + "/features.dat").c_str(),"w");
        for ( int i = 0; i < keyframes.size(); i++ )
        {
            int nfeatures = keyframes[i].features.size();
            fwrite(&nfeatures,sizeof(int),1,featuresf);
            for ( int j = 0; j < keyframes[i].features.size(); j++ )
            {
                //fwrite(&keyframes[i].features[j].x,sizeof(float),1,featuresf);
                //fwrite(&keyframes[i].features[j].y,sizeof(float),1,featuresf);
                //fwrite((float*)keyframes[i].features[j].descriptor.data,sizeof(float),128,featuresf);
                fwrite(&keyframes[i].features.points[j].x,sizeof(float),1,featuresf);
                fwrite(&keyframes[i].features.points[j].y,sizeof(float),1,featuresf);
                fwrite((float*)keyframes[i].features.descs.row(j).data,sizeof(float),128,featuresf);
            }
        }
        fclose(featuresf);

        FILE *matchesf = fopen((outputpath + "/matches.dat").c_str(),"w");
        int nmatches = image_matches.size();
        fwrite(&nmatches,sizeof(int),1,matchesf);
        for ( int i = 0; i < image_matches.size(); i++ )
        {
            const ImageMatch &m = image_matches[i];
            fwrite(&m.index0,sizeof(int),1,matchesf);
            fwrite(&m.index1,sizeof(int),1,matchesf);
            int nmatches = m.matches.size();
            fwrite(&nmatches,sizeof(int),1,matchesf);
            for ( Matches::const_iterator it = m.matches.begin(); it != m.matches.end(); it++ )
            {
                int first = it->first;
                int second = it->second;
                fwrite(&first,sizeof(int),1,matchesf);
                fwrite(&second,sizeof(int),1,matchesf);
            }
            fwrite(m.R.data(),sizeof(double),9,matchesf);
        }
        fclose(matchesf);

    }

    bool read_feature_tracks( const std::string &outputpath, std::vector<Keyframe> &keyframes, std::vector<ImageMatch> &image_matches )
    {
        std::string keyframespath = outputpath + "/keyframes.txt";
        FILE *keyframesf = fopen( keyframespath.c_str(), "r" );
        if ( !keyframesf ) return false;
        int nkeyframes;
        fscanf(keyframesf,"%d\n",&nkeyframes);
        std::vector<int> indices(nkeyframes);
        for ( int i = 0; i < nkeyframes; i++ )
        {
            fscanf(keyframesf,"%d ",&indices[i]);
        }
        fclose(keyframesf);
        std::cout << "read " << indices.size() << " indices\n";

        FILE *featuresf = fopen((outputpath + "/features.dat").c_str(),"r");
        for ( int i = 0; i < nkeyframes; i++ )
        {
            int nfeatures;
            fread(&nfeatures,sizeof(int),1,featuresf);
            Features features;
            for ( int j = 0; j < nfeatures; j++ )
            {
                float x, y;
                cv::Mat descriptor(1,128,CV_32F);
                fread(&x,sizeof(float),1,featuresf);
                fread(&y,sizeof(float),1,featuresf);
                fread((float*)descriptor.data,sizeof(float),128,featuresf);
                //features.push_back(Feature(x,y,descriptor));
                features.points.push_back(cv::Point2f(x,y));
                features.descs.push_back(descriptor);
            }
            char name[1024];
            sprintf(name,"%06d.jpg",indices[i]+1);
            keyframes.push_back(Keyframe(indices[i],name,features));
        }
        fclose(featuresf);

        FILE *matchesf = fopen((outputpath + "/matches.dat").c_str(),"r");
        int nmatches;
        fread(&nmatches,sizeof(int),1,matchesf);
        for ( int i = 0; i < nmatches; i++ )
        {
            int index0, index1;
            fread(&index0,sizeof(int),1,matchesf);
            fread(&index1,sizeof(int),1,matchesf);
            int nmatches;
            fread(&nmatches,sizeof(int),1,matchesf);
            Matches m;
            for ( int j = 0; j < nmatches; j++ )
            {
                int first, second;
                fread(&first,sizeof(int),1,matchesf);
                fread(&second,sizeof(int),1,matchesf);
                m[first] = second;
            }
            Eigen::Matrix3d R;
            fread(R.data(),sizeof(double),9,matchesf);
            image_matches.push_back(ImageMatch(index0,index1,m,R));
        }
        fclose(matchesf);
        return true;
    }
    
    void read_images( const std::string &videopath, std::vector<Keyframe> &keyframes )
    {
        std::cout << "loading images...\n";
        cv::VideoCapture cap(videopath);
        int video_index = -1;
        for ( int i = 0; i < keyframes.size(); i++ )
        {
            while ( video_index < keyframes[i].index )
            {
                cv::Mat image_in;
                if ( !cap.read(image_in) )
                {
                    std::cout << "could not read all keyframe images from " << videopath << "\n";
                    exit(1);
                }
                if ( image_in.channels() == 3 ) cv::cvtColor( image_in, keyframes[i].image, cv::COLOR_BGR2GRAY );
                else image_in.copyTo(keyframes[i].image);
                video_index++;
            }
        }
    }

}
