#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing/shape_predictor.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <opencv2/opencv.hpp>

using namespace dlib;
using namespace std;
//---------------------------------------------------------------------------

double euclideanDistance(const point& p1, const point& p2) {
    return sqrt(pow(p2.x() - p1.x(), 2) + pow(p2.y() - p1.y(), 2));
}

// Function to calculate Eye Aspect Ratio (EAR) using Euclidean distance
double calculateEAR(const full_object_detection& shape) {
    // Get the coordinates for the eye landmarks
    double V1 = euclideanDistance(shape.part(38), shape.part(40)); 
    cout << "V1:"<<V1<<endl;
    double V2 = euclideanDistance(shape.part(37), shape.part(41)); 
    cout << "V2:"<<V2<<endl;
    double H = euclideanDistance(shape.part(39), shape.part(36));
    cout << "H:"<<H<<endl;   
    
    double left= (V1 + V2) / (2.0 * H); // EAR formula
    
    double H1 = euclideanDistance(shape.part(43), shape.part(47)); 
    cout << "V1:"<<V1<<endl;
    double H2 = euclideanDistance(shape.part(44), shape.part(46)); 
    cout << "V2:"<<V2<<endl;
    double V = euclideanDistance(shape.part(42), shape.part(45));
    cout << "H:"<<H<<endl; 
    double right=(H1+H2)/(2.0*V);
    
    double EAR= (right + left )/2.0;
    return EAR;
    
}



//-------------------------------------------------------------------------------
/*
// Function to calculate Eye Aspect Ratio (EAR)
double calculateEAR(const full_object_detection& shape) {
    // Get the coordinates for the eye landmarks
    double B = length(shape.part(37) - shape.part(41)); 
    cout << "B:"<<B<<endl;
    double A = length(shape.part(38) - shape.part(40)); 
    cout << "A:"<<A<<endl;
    double C= length(shape.part(39) - shape.part(36));
    cout << "C:"<<C<<endl;
    double left=(A+B)/ (2.0 * C); // EAR formula
    cout << "left_eye:" <<left<<endl;
    
    double  X= length(shape.part(43) - shape.part(47)); 
    cout << "X:"<<C<<endl;
    double  Y= length(shape.part(44) - shape.part(46)); 
    cout << "Y:"<<C<<endl;
    double Z= length(shape.part(45) - shape.part(42));
    cout << "Z:"<<C<<endl;
    double right= (X+Y)/ (2.0 * Z); // EAR formula
    cout << "right_eye:" <<right<<endl;
    
    double EAR=(left+right) / 2.0;
    cout <<"EAR :"<<EAR <<endl;
    return EAR;
}
*/

// Function to calculate Mouth Aspect Ratio (MAR)
double calculateMAR(const full_object_detection& shape) {
    // Get the coordinates for the mouth landmarks
    /*
    double C = euclideanDistance(shape.part(49), shape.part(60)); // Vertical distance between mouth landmarks
    double D = euclideanDistance(shape.part(50), shape.part(59)); // Horizontal distance between mouth landmarks
    double E = euclideanDistance(shape.part(51), shape.part(58));
    double F = euclideanDistance(shape.part(51), shape.part(56));
    double G = euclideanDistance(shape.part(53), shape.part(55));
    double H = euclideanDistance(shape.part(48), shape.part(54));
    double mar = (C - D - E - F - G) / (2.0 * H);
    return mar < 0 ? 0 : mar;*/
    
     dlib::point p62 = shape.part(62);
    dlib::point p64 = shape.part(64);
    dlib::point p66 = shape.part(66);
    dlib::point p58 = shape.part(58);
    dlib::point p60 = shape.part(60);
    dlib::point p68 = shape.part(68);

    // Calculate distances
    double A = euclideanDistance(p62, p66); // Vertical distance
    double B = euclideanDistance(p64, p60); // Vertical distance
    double C = euclideanDistance(shape.part(61), shape.part(65)); // Horizontal distance

    // Check for zero division
    if (C == 0) {
        cerr << "Error: Horizontal distance C is zero, cannot compute MAR." << endl;
        return -1; // Or some error code
    }

    // Calculate MAR
    return (A + B) / (2.0 * C);

}
int main() {
    try {
        // Load face detection and shape prediction models
        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor sp;
        deserialize("/home/itstarkenn/opencv_practice/face_dlib/demo/shape_predictor_68_face_landmarks.dat") >> sp;

        // Load input image
        array2d<rgb_pixel> img;
        load_image(img, "/home/itstarkenn/Downloads/download (3).jpeg");

        // Detect faces in the image
        std::vector<rectangle> dets = detector(img);
        cout << "Number of faces detected: " << dets.size() << endl;

        // Create a window to display the image with landmarks
        image_window win;

        // Draw all bounding boxes first
        win.set_image(img);  // Set the image before adding overlays
        for (const auto& det : dets) {
            win.add_overlay(det, rgb_pixel(255, 0, 0)); // Red color for bounding box
        }

        // Iterate through each detected face for landmarks
        for (size_t i = 0; i < dets.size(); ++i) {
            // Detect landmarks for this face
            full_object_detection shape = sp(img, dets[i]);

            // Draw landmarks on the image
            for (size_t j = 0; j < shape.num_parts(); ++j) {
                int x = shape.part(j).x();
                int y = shape.part(j).y();
                draw_solid_circle(img, point(x, y), 2, rgb_pixel(0, 255, 0)); // Green color for landmarks
                
std::string text = std::to_string(j);
draw_string(img, point(x + 5, y - 5), text, rgb_pixel(255, 255, 255)); // White color for text
            }

	   
            // Calculate EAR and MAR
           // double ear = calculateEAR(shape);
            double mar = calculateMAR(shape);
           //cout << "Eye Aspect Ratio (EAR): " << ear << endl;
           cout << "Mouth Aspect Ratio (MAR): " << mar << endl;


             double ear = calculateEAR(shape);
            cout << "Eye Aspect Ratio (EAR): " << ear << endl;
            // Display image with landmarks and bounding boxes
            win.set_image(img);  // Update the window with the modified image
            win.add_overlay(render_face_detections(shape));  // Add landmark overlays

            cout << "Number of landmarks: " << shape.num_parts() << endl;

            // Print landmark points
            for (size_t j = 0; j < shape.num_parts(); ++j) {
                cout << "Landmark #" << j << ": " << shape.part(j) << endl;
            }
            
        }

        // Save the image with landmarks and bounding boxes
        save_png(img, "output.png");
        cout << "Press Enter to exit..." << endl;
    std::cin.get(); 

    } catch (std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
    }
    return 0;
}
