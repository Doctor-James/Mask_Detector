#include "Mask_Detection.h"
using namespace cv;
Mask_Detection::Mask_Detection()
{
    std::string engine_name = "../engine/mask_100epoch_v5.0.engine";
    engine_init(engine_name);
    cap.open(0);
}

void Mask_Detection::process()
{
    while(1)
    {
        cap>>srcImage;
        if(srcImage.empty())
        {
            continue;
        }
        yolo_main(srcImage);
        namedWindow("srcImage",0);
        imshow("srcImage",srcImage);
        if(waitKey(10) == 'q')
        {
            break;
        }

    }
}

int main()
{
    Mask_Detection Mask_Detection_;
    Mask_Detection_.process();
}