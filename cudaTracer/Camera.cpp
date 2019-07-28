#include "Camera.h"
#include "globals.h"

float Camera::planeNormMult = 1;
float Camera::planeNormDistMult = 1;

Camera::Camera() {
    vecPosition[Vec3f::X] = 0.0f;
    vecPosition[Vec3f::Y] = 0.0f;
    vecPosition[Vec3f::Z] = 0.0f;

    vecUp[Vec3f::X] = 0.0f;
    vecUp[Vec3f::Y] = 1.0f;
    vecUp[Vec3f::Z] = 0.0f;

    vecLookAt[Vec3f::X] = 0.0f;
    vecLookAt[Vec3f::Y] = 0.0f;
    vecLookAt[Vec3f::Z] = 1.0f;

    vecRight[Vec3f::X] = 1.0f;
    vecRight[Vec3f::Y] = 0.0f;
    vecRight[Vec3f::Z] = 0.0f;

    calcView();

    // halfFovAsRadians = (float)ToRadian(p_fovInDegrees / 2.0f);
    // aspectRatio = p_aspectRatio;
    // nearDist = p_near;
    // farDist = p_far;

    // calcProjection();

    //// Debug Vars
    // DebugGUI* dGui = DebugGUI::getInstance();
    // dGui->addVar( "FrustumCulling", DebugGUI::DG_FLOAT, DebugGUI::READ_WRITE,
    // "Camera NormMult", 	&planeNormMult, "group='Plane settings'" );
    // dGui->addVar( "FrustumCulling", DebugGUI::DG_FLOAT, DebugGUI::READ_WRITE,
    // "Camera NormDistMult", 	&planeNormDistMult, "group='Plane settings'" );

    // TODO: debug
    DebugGUI* dg = ServiceRegistry::getInstance().get<DebugGUI>();
    dg->addVar("Camera", DebugGUI::DG_VEC3, DebugGUI::READ_WRITE, "vecUp",
               &(vecUp));
    dg->addVar("Camera", DebugGUI::DG_VEC3, DebugGUI::READ_WRITE, "vecLookAt",
               &(vecLookAt));
    dg->addVar("Camera", DebugGUI::DG_VEC3, DebugGUI::READ_WRITE, "vecRight",
               &(vecRight));
    dg->addVar("Camera", DebugGUI::DG_VEC3, DebugGUI::READ_WRITE, "vecPosition",
               &(vecPosition));
    dg->addVar("Camera", DebugGUI::DG_VEC3, DebugGUI::READ_WRITE, "View 1",
               &reinterpret_cast<float*>(&matView)[0]);
    dg->addVar("Camera", DebugGUI::DG_VEC3, DebugGUI::READ_WRITE, "View 2",
               &reinterpret_cast<float*>(&matView)[4]);
    dg->addVar("Camera", DebugGUI::DG_VEC3, DebugGUI::READ_WRITE, "View 3",
               &reinterpret_cast<float*>(&matView)[8]);
    dg->addVar("Camera", DebugGUI::DG_VEC3, DebugGUI::READ_WRITE, "View 4",
               &reinterpret_cast<float*>(&matView)[12]);
}

void Camera::update() {
    // Normalize axes
    // D3DXVec3Normalize(&vecLookAt, &vecLookAt);
    vecLookAt = vecLookAt.normalized();

    // D3DXVec3Cross(&vecUp, &vecLookAt, &vecRight);
    vecUp = vecLookAt.cross(vecRight);
    // D3DXVec3Normalize(&vecUp, &vecUp);
    vecUp = vecUp.normalized();

    // D3DXVec3Cross(&vecRight, &vecUp, &vecLookAt);
    vecRight = vecUp.cross(vecLookAt);
    // D3DXVec3Normalize(&vecRight, &vecRight);
    vecRight = vecRight.normalized();

    calcView();
    // matFinal = matView * matProjection;
}

Mat44f Camera::getCamera() { return matView; }

void Camera::getXZ(float* p_x, float* p_z) {
    *p_x = vecPosition[Vec3f::X];
    *p_z = vecPosition[Vec3f::Z];
}

Vec3f Camera::getPosition() { return vecPosition; }

/*Frustum Camera::getFrustum()
{
        float a[Frustum::NUM_PLANES];
        float b[Frustum::NUM_PLANES];
        float c[Frustum::NUM_PLANES];
        float d[Frustum::NUM_PLANES];

        a[Frustum::F_LEFT]		= matFinal._14 + matFinal._11;
        b[Frustum::F_LEFT]		= matFinal._24 + matFinal._21;
        c[Frustum::F_LEFT]		= matFinal._34 + matFinal._31;
        d[Frustum::F_LEFT]		= matFinal._44 + matFinal._41;

        a[Frustum::F_RIGHT]		= matFinal._14 - matFinal._11;
        b[Frustum::F_RIGHT]		= matFinal._24 - matFinal._21;
        c[Frustum::F_RIGHT]		= matFinal._34 - matFinal._31;
        d[Frustum::F_RIGHT]		= matFinal._44 - matFinal._41;

        a[Frustum::F_BOTTOM]	= matFinal._14 + matFinal._12;
        b[Frustum::F_BOTTOM]	= matFinal._24 + matFinal._22;
        c[Frustum::F_BOTTOM]	= matFinal._34 + matFinal._32;
        d[Frustum::F_BOTTOM]	= matFinal._44 + matFinal._42;

        a[Frustum::F_TOP]		= matFinal._14 - matFinal._12;
        b[Frustum::F_TOP]		= matFinal._24 - matFinal._22;
        c[Frustum::F_TOP]		= matFinal._34 - matFinal._32;
        d[Frustum::F_TOP]		= matFinal._44 - matFinal._42;

        a[Frustum::F_NEAR]		= matFinal._13;
        b[Frustum::F_NEAR]		= matFinal._23;
        c[Frustum::F_NEAR]		= matFinal._33;
        d[Frustum::F_NEAR]		= matFinal._43;

        a[Frustum::F_FAR]		= matFinal._14 - matFinal._13;
        b[Frustum::F_FAR]		= matFinal._24 - matFinal._23;
        c[Frustum::F_FAR]		= matFinal._34 - matFinal._33;
        d[Frustum::F_FAR]		= matFinal._44 - matFinal._43;

        Frustum frustum;
        for(int i=0; i<Frustum::NUM_PLANES; i++)
        {
                frustum.planes[i] = Plane(
                        planeNormMult*a[i],
                        planeNormMult*b[i],
                        planeNormMult*c[i],
                        planeNormDistMult*d[i]);
        }

        return frustum;
}*/

Vec3f Camera::getRight() { return vecRight; }

/*void Camera::setNearAndFar( float p_near, float p_far )
{
        nearDist = p_near;
        farDist = p_far;
        calcProjection();
}
*/

/*
void Camera::calcProjection()
{
        //D3DXMatrixPerspectiveFovLH(&matProjection, halfFovAsRadians,
aspectRatio, nearDist, farDist);

        D3DXMATRIX persp;
        ZeroMemory(persp, sizeof(D3DXMATRIX));

        persp._11 = 1/(aspectRatio*(tan(halfFovAsRadians/2)));
        persp._22 = 1/(tan(halfFovAsRadians/2));
        persp._33 = farDist/(farDist - nearDist);
        persp._34 = 1.0f;
        persp._43 = (-nearDist * farDist)/(farDist - nearDist);

        matProjection = persp;
        //matInvProjection = D3DXMatrixInverse(&persp,
D3DXMatrixDeterminant(&persp),&persp);
}
*/

void Camera::calcView() {
    float x = -vecPosition.dot(vecRight);
    float y = -vecPosition.dot(vecUp);
    float z = -vecPosition.dot(vecLookAt);

    matView[0][0] = vecRight[Vec3f::X];
    matView[1][0] = vecRight[Vec3f::Y];
    matView[2][0] = vecRight[Vec3f::Z];
    matView[3][0] = vecPosition[Vec3f::X];

    matView[0][1] = vecUp[Vec3f::X];
    matView[1][1] = vecUp[Vec3f::Y];
    matView[2][1] = vecUp[Vec3f::Z];
    matView[3][1] = vecPosition[Vec3f::Y];

    matView[0][2] = vecLookAt[Vec3f::X];
    matView[1][2] = vecLookAt[Vec3f::Y];
    matView[2][2] = vecLookAt[Vec3f::Z];
    matView[3][2] = vecPosition[Vec3f::Z];

    matView[0][3] = 0.0f;
    matView[1][3] = 0.0f;
    matView[2][3] = 0.0f;
    matView[3][3] = 1.0f;

    ray = matView./*inversed().*/ multVec(ray);
}

void Camera::setPosition(float p_x, float p_y, float p_z) {
    vecPosition[Vec3f::X] = p_x;
    vecPosition[Vec3f::Y] = p_y;
    vecPosition[Vec3f::Z] = p_z;
}

void Camera::setPosition(Vec3f p_pos) { vecPosition = p_pos; }

void Camera::setLookAt(Vec3f p_lookAt) {
    vecUp = Vec3f(0.0f, 1.0f, 0.0f);
    vecLookAt = p_lookAt - vecPosition;
    // D3DXVec3Cross(&vecRight, &vecUp, &vecLookAt);
    vecRight = vecUp.cross(vecLookAt);
    // D3DXVec3Cross(&vecUp, &vecLookAt, &vecRight);
    vecUp = vecLookAt.cross(vecRight);
    // D3DXVec3Normalize( &vecRight, &vecRight );
    vecRight = vecRight.normalized();
    // D3DXVec3Normalize( &vecUp, &vecUp );
    vecUp = vecUp.normalized();
    // D3DXVec3Normalize( &vecLookAt, &vecLookAt );
    vecLookAt = vecLookAt.normalized();
    update();
}

void Camera::setDirection(Vec3f p_direction) {
    vecUp = Vec3f(0.0f, 1.0f, 0.0f);
    vecLookAt = p_direction;
    // D3DXVec3Cross(&vecRight, &vecUp, &vecLookAt);
    vecRight = vecUp.cross(vecLookAt);
    // D3DXVec3Cross(&vecUp, &vecLookAt, &vecRight);
    vecUp = vecLookAt.cross(vecRight);
    // D3DXVec3Normalize( &vecRight, &vecRight );
    vecRight = vecRight.normalized();
    // D3DXVec3Normalize( &vecUp, &vecUp );
    vecUp = vecUp.normalized();
    // D3DXVec3Normalize( &vecLookAt, &vecLookAt );
    vecLookAt = vecLookAt.normalized();
    update();
}

void Camera::setZ(float p_z) { vecPosition[Vec3f::Z] = p_z; }

void Camera::strafe(float p_amount) { vecPosition += p_amount * vecRight; }

void Camera::walk(float p_amount) { vecPosition += p_amount * vecLookAt; }

void Camera::ascend(float p_amount) { vecPosition += p_amount * vecUp; }


void Camera::pitch(float p_angle) {
    // D3DXMATRIX _matRotation;
    // D3DXMatrixRotationAxis(&_matRotation, &vecRight, p_angle/p_dt);
    Mat44f rotMat = Mat44f::rotation(vecRight, p_angle);

    // D3DXVec3TransformNormal(&vecUp, &vecUp, &_matRotation);
    vecUp = rotMat.multVec(vecUp);
    // D3DXVec3TransformNormal(&vecLookAt, &vecLookAt, &_matRotation);
     vecLookAt = rotMat.multVec(vecLookAt);
}

void Camera::rotateY(float p_angle) {
    // D3DXMATRIX _matRotation;
    // D3DXMatrixRotationY(&_matRotation, p_angle/p_dt);
    Mat44f rotMat = Mat44f::rotationY(p_angle);

    // D3DXVec3TransformNormal(&vecRight, &vecRight, &_matRotation);
    vecRight = rotMat.multVec(vecRight);
    // D3DXVec3TransformNormal(&vecUp, &vecUp, &_matRotation);
    vecUp = rotMat.multVec(vecUp);
    // D3DXVec3TransformNormal(&vecLookAt, &vecLookAt, &_matRotation);
    vecLookAt = rotMat.multVec(vecLookAt);
}