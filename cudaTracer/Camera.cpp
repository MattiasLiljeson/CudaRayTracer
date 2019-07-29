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

    // TODO: debug
    DebugGUI* dg = ServiceRegistry::instance().get<DebugGUI>();
    dg->setSize("Camera", 200, 1400);
    dg->setPosition("Camera", 0, 0);
    dg->setVisible("Camera", false);
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
    vecLookAt = vecLookAt.normalized();
    vecUp = vecLookAt.cross(vecRight);
    vecUp = vecUp.normalized();
    vecRight = vecUp.cross(vecLookAt);
    vecRight = vecRight.normalized();
    calcView();
}

Mat44f Camera::getCamera() { return matView; }

void Camera::getXZ(float* p_x, float* p_z) {
    *p_x = vecPosition[Vec3f::X];
    *p_z = vecPosition[Vec3f::Z];
}

Vec3f Camera::getPosition() { return vecPosition; }

Vec3f Camera::getRight() { return vecRight; }

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

    ray = matView.multVec(ray);
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
    vecRight = vecUp.cross(vecLookAt);
    vecUp = vecLookAt.cross(vecRight);
    vecRight = vecRight.normalized();
    vecUp = vecUp.normalized();
    vecLookAt = vecLookAt.normalized();
    update();
}

void Camera::setDirection(Vec3f p_direction) {
    vecUp = Vec3f(0.0f, 1.0f, 0.0f);
    vecLookAt = p_direction;
    vecRight = vecUp.cross(vecLookAt);
    vecUp = vecLookAt.cross(vecRight);
    vecRight = vecRight.normalized();
    vecUp = vecUp.normalized();
    vecLookAt = vecLookAt.normalized();
    update();
}

void Camera::setZ(float p_z) { vecPosition[Vec3f::Z] = p_z; }

void Camera::strafe(float p_amount) { vecPosition += p_amount * vecRight; }

void Camera::walk(float p_amount) { vecPosition += p_amount * vecLookAt; }

void Camera::ascend(float p_amount) { vecPosition += p_amount * vecUp; }


void Camera::pitch(float p_angle) {
    Mat44f rotMat = Mat44f::rotation(vecRight, p_angle);
    vecUp = rotMat.multVec(vecUp);
     vecLookAt = rotMat.multVec(vecLookAt);
}

void Camera::rotateY(float p_angle) {
    Mat44f rotMat = Mat44f::rotationY(p_angle);
    vecRight = rotMat.multVec(vecRight);
    vecUp = rotMat.multVec(vecUp);
    vecLookAt = rotMat.multVec(vecLookAt);
}