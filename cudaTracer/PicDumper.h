#pragma once

#include <lodepng.h>
#include <ctime>

#include "DeviceHandler.h"
#include "Service.h"
#include "TextureRenderer.h"

class PicDumper : public Service {
    int picWidth;
    int picHeight;

   public:
    PicDumper(int picWidth, int picHeight) {
        this->picWidth = picWidth;
        this->picHeight = picHeight;
    }

    void update(float dt) {
        InputHandler* input = ServiceRegistry::instance().get<InputHandler>();
        if (input->getKey(InputHandler::F2)) {
            TextureRenderer* tr =
                ServiceRegistry::instance().get<TextureRenderer>();
            toDisk(tr);
        }
    }

   private:
    void toDisk(TextureRenderer* texRender) {
        std::time_t t = std::time(0);  // t is an integer type
        char buf[128];
        sprintf(buf, "%ix%i-%i.png", picWidth, picHeight, t);
        string timerName(buf);

        unsigned int picSize = picWidth * 4 * picHeight;
        vector<float> arr(picSize);
        texRender->copyToHostArray(&arr[0]);
        vector<unsigned char> img = copyFloatsToCharVector(arr);
        unsigned error = lodepng::encode(buf, img, picWidth, picHeight);
    }

    vector<unsigned char> copyFloatsToCharVector(vector<float>& pixels) {
        vector<unsigned char> img(pixels.size());
        for (unsigned int i = 0; i < pixels.size(); i++) {
            int tmp = (int)(pixels[i] * 256.0f);
            if (tmp > 255) {
                img[i] = 255;
            } else if (tmp < 0) {
                img[i] = 0;
            } else {
                img[i] = (unsigned char)tmp;
            }
        }
        return img;
    }
};