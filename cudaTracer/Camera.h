#ifndef CAMERA_H
#define CAMERA_H

#include <cmath>

//#include <d3d10.h>
//#include <d3dx10.h>

//#include "Frustum.h"
#include "DebugGUI.h"
#include "Mat.cuh"
#include "Utils.h"
#include "Vec.cuh"

class Camera {
   private:
    Mat44f matView;
    //Mat44f matProjection;
    //Mat44f matFinal;

    Vec3f vecPosition;
    Vec3f vecLookAt;
    Vec3f vecUp;
    Vec3f vecRight;

    Vec3f ray;

    //float halfFovAsRadians;
    //float aspectRatio;
    //float nearDist;
    //float farDist;

    // Debug vars shared by all Cameras
    static float planeNormMult;
    static float planeNormDistMult;

   public:
    Camera();
    void update();
    Mat44f getCamera();
    //void calcProjection();
    void calcView();
    void getXZ(float* p_x, float* p_z);
    Vec3f getPosition();
    // Frustum getFrustum();
    Vec3f getRight();
    //void setNearAndFar(float p_near, float p_far);
    void setPosition(Vec3f p_pos);
    void setPosition(float p_x, float p_y, float p_z);
    void setLookAt(Vec3f p_lookAt);
    void setDirection(Vec3f p_direction);
    void setZ(float p_z);

    void strafe(float p_amount);
    void walk(float p_amount);
    void ascend(float p_amount);

    void pitch(float p_angle);
    void rotateY(float p_angle);
};

#endif
