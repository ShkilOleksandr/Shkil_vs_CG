// Sources

// https://docs.gtk.org/gtk3/
// https://docs.gtk.org/gtk3/method.Box.pack_start.html
// https://docs.gtk.org/gtk3/method.Box.pack_end.html
// https://docs.gtk.org/gtk3/class.FileChooserDialog.html
// https://docs.gtk.org/gtk3/class.Paned.html

// https://docs.opencv.org/4.x/
// https://docs.opencv.org/4.x/db/deb/tutorial_display_image.html
// https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html
// https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html


#include <gtk/gtk.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <functional>
#include <math.h>
#include <fstream>
#include <cstdlib>


cv::Mat image, original_image;
GtkWidget *image_area;
GtkWidget *scrolled_window; 

int dbrightness = 20;
double gamma_coeff = 3.0;
int contrast_center = 127;
double contrast_coeff = 2.0;

int kernel_size = 3;
double sigma = 1.0;
double a = -1;
double b = kernel_size * kernel_size;


// Line drawing

std::vector<GdkPoint> points = {{0, 255}, {255, 0}};
GtkWidget *drawing_area;
const int threshold = 20;


// Pixelize

int pixelize_size = 100;

// Dithering

int num_shades = 2;


void apply_filter(std::function<cv::Vec3b(const cv::Mat&, int, int)> filter) {

    if (image.empty()) return;



    cv::Mat new_image = image.clone(); 



    for (int y = 1; y < image.rows - 1; y++) {  

        for (int x = 1; x < image.cols - 1; x++) {

            new_image.at<cv::Vec3b>(y, x) = filter(image, x, y);

        }

    }



    image = new_image; 



    GdkPixbuf *pixbuf = gdk_pixbuf_new_from_data(

        image.data, GDK_COLORSPACE_RGB, FALSE, 8,

        image.cols, image.rows, image.step, NULL, NULL

    );

    gtk_image_set_from_pixbuf(GTK_IMAGE(image_area), pixbuf);

}

void apply_inversion(GtkWidget *widget, gpointer data) {

    apply_filter([](const cv::Mat &img, int x, int y) -> cv::Vec3b {

        cv::Vec3b pixel = img.at<cv::Vec3b>(y, x);

        return cv::Vec3b(255 - pixel[0], 255 - pixel[1], 255 - pixel[2]);

    });

}

void apply_more_brightness(GtkWidget *widget, gpointer data) {

    apply_filter([](const cv::Mat &img, int x, int y) -> cv::Vec3b {

        cv::Vec3b pixel = img.at<cv::Vec3b>(y, x);

        return cv::Vec3b(std::min(pixel[0] + dbrightness, 255), 

                         std::min(pixel[1] + dbrightness, 255), 

                         std::min(pixel[2] + dbrightness, 255));

    });

}

void apply_less_brightness(GtkWidget *widget, gpointer data) {

    apply_filter([](const cv::Mat &img, int x, int y) -> cv::Vec3b {

        cv::Vec3b pixel = img.at<cv::Vec3b>(y, x);

        return cv::Vec3b(std::max(pixel[0] - dbrightness, 0), 

                         std::max(pixel[1] - dbrightness, 0), 

                         std::max(pixel[2] - dbrightness, 0));

    });

}

void apply_gamma_bright(GtkWidget *widget, gpointer data) {

    apply_filter([](const cv::Mat &img, int x, int y) -> cv::Vec3b {

        cv::Vec3b pixel = img.at<cv::Vec3b>(y, x);

        return cv::Vec3b(

            std::floor(255 * pow(static_cast<double>(pixel[0]) / 255.0, 1.0 / gamma_coeff)),

            std::floor(255 * pow(static_cast<double>(pixel[1]) / 255.0, 1.0 / gamma_coeff)),

            std::floor(255 * pow(static_cast<double>(pixel[2]) / 255.0, 1.0 / gamma_coeff))

        );

    });

}

void apply_gamma_dark(GtkWidget *widget, gpointer data) {

    apply_filter([](const cv::Mat &img, int x, int y) -> cv::Vec3b {

        cv::Vec3b pixel = img.at<cv::Vec3b>(y, x);

        return cv::Vec3b(

            std::ceil(255 * pow(static_cast<double>(pixel[0]) / 255.0, gamma_coeff)),

            std::ceil(255 * pow(static_cast<double>(pixel[1]) / 255.0, gamma_coeff)),

            std::ceil(255 * pow(static_cast<double>(pixel[2]) / 255.0, gamma_coeff))

        );

    });

}

void apply_contrast(GtkWidget *widget, gpointer data) {

    apply_filter([](const cv::Mat &img, int x, int y) -> cv::Vec3b {

        cv::Vec3b pixel = img.at<cv::Vec3b>(y, x);

        return cv::Vec3b(

            static_cast<uchar>(std::max(std::min(std::ceil((static_cast<double>(pixel[0]) - contrast_center) * contrast_coeff + contrast_center), 255.0), 0.0)),

            static_cast<uchar>(std::max(std::min(std::ceil((static_cast<double>(pixel[1]) - contrast_center) * contrast_coeff + contrast_center), 255.0), 0.0)),

            static_cast<uchar>(std::max(std::min(std::ceil((static_cast<double>(pixel[2]) - contrast_center) * contrast_coeff + contrast_center), 255.0), 0.0))

        );

    });

}

void apply_blur(GtkWidget *widget, gpointer data) {

    apply_filter([](const cv::Mat &img, int x, int y) -> cv::Vec3b {

        cv::Vec3i sum(0, 0, 0); 

        for (int y_counter = -kernel_size / 2; y_counter <= kernel_size / 2; y_counter++) {

            for (int x_counter = -kernel_size / 2; x_counter <= kernel_size / 2; x_counter++) {



                sum += img.at<cv::Vec3b>(std::min(std::max(y + y_counter, 0), img.rows - 1), 

                                         std::min(std::max(x + x_counter, 0), img.cols - 1));



            }

        }

        return sum / (kernel_size * kernel_size);

    });

}

void apply_pixelize(int pixelize_size) {
    if (image.empty()) return;

    cv::Mat new_image = image.clone();

    for (int y = 0; y < image.rows; y += pixelize_size) {
        for (int x = 0; x < image.cols; x += pixelize_size) {

            cv::Vec3i sum(0, 0, 0);
            int count = 0;

            for (int dy = 0; dy < pixelize_size && (y + dy) < image.rows; ++dy) {
                for (int dx = 0; dx < pixelize_size && (x + dx) < image.cols; ++dx) {
                    sum += image.at<cv::Vec3b>(y + dy, x + dx);
                    ++count;
                }
            }

            cv::Vec3b avg_color = sum / count;

            for (int dy = 0; dy < pixelize_size && (y + dy) < image.rows; ++dy) {
                for (int dx = 0; dx < pixelize_size && (x + dx) < image.cols; ++dx) {
                    new_image.at<cv::Vec3b>(y + dy, x + dx) = avg_color;
                }
            }
        }
    }

    image = new_image;

    GdkPixbuf *pixbuf = gdk_pixbuf_new_from_data(
        image.data, GDK_COLORSPACE_RGB, FALSE, 8,
        image.cols, image.rows, image.step, NULL, NULL
    );
    gtk_image_set_from_pixbuf(GTK_IMAGE(image_area), pixbuf);
}

void pixelize(GtkWidget *widget, gpointer data) {
    
    apply_pixelize(pixelize_size);
}

void apply_greyscale(GtkWidget *widget, gpointer data) {
    apply_filter([](const cv::Mat &img, int x, int y) -> cv::Vec3b {
        cv::Vec3b pixel = img.at<cv::Vec3b>(y, x);
        
        uchar grey_value = static_cast<uchar>(
            0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]);

        return cv::Vec3b(grey_value, grey_value, grey_value);
    });
}

void apply_random_dithering(GtkWidget *widget, gpointer data) {
    apply_filter([](const cv::Mat &img, int x, int y) -> cv::Vec3b {
        cv::Vec3b pixel = img.at<cv::Vec3b>(y, x);

        bool is_greyscale = (pixel[0] == pixel[1] && pixel[1] == pixel[2]);

        int levels = num_shades;
        if (levels < 2) levels = 2;
        int step = 255 / (levels - 1); 

        if (is_greyscale) {
            uchar grey_value = pixel[0];

            int threshold = rand() % step; 

            int lower_level = (grey_value / step) * step;
            int upper_level = std::min(lower_level + step, 255);

            uchar dithered_value = (grey_value % step > threshold) ? upper_level : lower_level;

            return cv::Vec3b(dithered_value, dithered_value, dithered_value);
        } else {
            cv::Vec3b dithered_pixel;
            for (int c = 0; c < 3; c++) {
                int threshold = rand() % step;

                int lower_level = (pixel[c] / step) * step;
                int upper_level = std::min(lower_level + step, 255);

                dithered_pixel[c] = (pixel[c] % step > threshold) ? upper_level : lower_level;
            }
            return dithered_pixel;
        }
    });
}

void update_num_shades(GtkWidget *widget, gpointer data) {
    num_shades = static_cast<int>(gtk_range_get_value(GTK_RANGE(widget)));
}

GtkWidget* create_labeled_shades_slider() {
    GtkWidget *slider_container = gtk_box_new(GTK_ORIENTATION_VERTICAL, 2); 

    GtkWidget *label = gtk_label_new("Number of Color values per channel:");
    gtk_widget_set_halign(label, GTK_ALIGN_CENTER);

    GtkWidget *scale = gtk_scale_new_with_range(GTK_ORIENTATION_HORIZONTAL, 2, 255, 1);
    gtk_range_set_value(GTK_RANGE(scale), num_shades);
    gtk_scale_set_digits(GTK_SCALE(scale), 0);
    gtk_scale_set_draw_value(GTK_SCALE(scale), TRUE);
    gtk_widget_set_margin_bottom(scale, 5);

    g_signal_connect(scale, "value-changed", G_CALLBACK(update_num_shades), NULL);

    gtk_box_pack_start(GTK_BOX(slider_container), label, FALSE, FALSE, 2);
    gtk_box_pack_start(GTK_BOX(slider_container), scale, FALSE, FALSE, 2);

    return slider_container;
}

cv::Mat generate_gauss_kernel(int size, double sigma) {

    cv::Mat kernel(size, size, CV_64F);

    double sum = 0.0;

    int half_size = size / 2;



    for (int y = -half_size; y <= half_size; y++) {

        for (int x = -half_size; x <= half_size; x++) {

            double value = (1.0 / (2.0 * M_PI * sigma * sigma)) * 

                           exp(-(x * x + y * y) / (2.0 * sigma * sigma));

            kernel.at<double>(y + half_size, x + half_size) = value;

            sum += value;

        }

    }

    kernel /= sum;



    return kernel;

}

cv::Mat generate_sharpen_kernel(int size) {

    cv::Mat kernel(size, size, CV_64F);

    int half_size = size / 2;



    for (int y = -half_size; y <= half_size; y++) {

        for (int x = -half_size; x <= half_size; x++) {

            kernel.at<double>(y + half_size, x + half_size) = a;



        }

    }

    kernel.at<double>(half_size, half_size) = b;



    return kernel;

}

cv::Mat generate_diagonal_kernel(int size) {

    cv::Mat kernel = cv::Mat::zeros(size, size, CV_64F); 



    for (int i = 0; i < size - 1; i++) {

        kernel.at<double>(i, i) = -1;   

    }



    kernel.at<double>(size / 2, size / 2) = 1;  



    return kernel;

}

cv::Mat generate_southemboss_kernel(int size) {

    cv::Mat kernel = cv::Mat::zeros(size, size, CV_64F); 

    int half_size = size / 2;



    for (int y = 0; y < half_size; y++) {

        for (int x = 0; x < size; x++) {

            kernel.at<double>(y, x) = -1;

        }

    }



    kernel.at<double>(half_size, half_size) = 1;



    for (int y = half_size + 1; y < size; y++) {

        for (int x = 0; x < size; x++) {

            kernel.at<double>(y, x) = 1;

        }

    }



    return kernel;

}

void apply_gaussblur(GtkWidget *widget, gpointer data) {



    cv::Mat kernel2D = generate_gauss_kernel(kernel_size, sigma);



    apply_filter([kernel2D](const cv::Mat &img, int x, int y) -> cv::Vec3b {

        cv::Vec3d sum(0, 0, 0); 

        for (int y_counter = -kernel_size / 2; y_counter <= kernel_size / 2; y_counter++) {

            for (int x_counter = -kernel_size / 2; x_counter <= kernel_size / 2; x_counter++) {



                double weight = kernel2D.at<double>(y_counter + kernel_size / 2, x_counter + kernel_size / 2);

                sum += weight * static_cast<cv::Vec3d>(img.at<cv::Vec3b>(std::min(std::max(y + y_counter, 0), img.rows - 1),

                                                                         std::min(std::max(x + x_counter, 0), img.cols - 1)));

            }

        }

        return cv::Vec3b(

            cv::saturate_cast<uchar>(std::ceil(sum[0])),

            cv::saturate_cast<uchar>(std::ceil(sum[1])),

            cv::saturate_cast<uchar>(std::ceil(sum[2]))

        );

    });

}

void apply_sharpen(GtkWidget *widget, gpointer data) {



    cv::Mat kernel2D = generate_sharpen_kernel(kernel_size);



    apply_filter([kernel2D](const cv::Mat &img, int x, int y) -> cv::Vec3b {

        cv::Vec3d sum(0, 0, 0); 

        for (int y_counter = -kernel_size / 2; y_counter <= kernel_size / 2; y_counter++) {

            for (int x_counter = -kernel_size / 2; x_counter <= kernel_size / 2; x_counter++) {



                double weight = kernel2D.at<double>(y_counter + kernel_size / 2, x_counter + kernel_size / 2);

                sum += weight * static_cast<cv::Vec3d>(img.at<cv::Vec3b>(std::min(std::max(y + y_counter, 0), img.rows - 1),

                                                                         std::min(std::max(x + x_counter, 0), img.cols - 1)));

            }

        }

        return cv::Vec3b(

            cv::saturate_cast<uchar>(sum[0]),

            cv::saturate_cast<uchar>(sum[1]),

            cv::saturate_cast<uchar>(sum[2])

        );

    });

}

void apply_edges(GtkWidget *widget, gpointer data) {



    cv::Mat kernel2D = generate_diagonal_kernel(kernel_size);



    apply_filter([kernel2D](const cv::Mat &img, int x, int y) -> cv::Vec3b {

        cv::Vec3d sum(0, 0, 0); 

        for (int y_counter = -kernel_size / 2; y_counter <= kernel_size / 2; y_counter++) {

            for (int x_counter = -kernel_size / 2; x_counter <= kernel_size / 2; x_counter++) {



                double weight = kernel2D.at<double>(y_counter + kernel_size / 2, x_counter + kernel_size / 2);

                sum += weight * static_cast<cv::Vec3d>(img.at<cv::Vec3b>(std::min(std::max(y + y_counter, 0), img.rows - 1),

                                                                         std::min(std::max(x + x_counter, 0), img.cols - 1)));

            }

        }

        return cv::Vec3b(

            cv::saturate_cast<uchar>(sum[0]),

            cv::saturate_cast<uchar>(sum[1]),

            cv::saturate_cast<uchar>(sum[2])

        );

    });

}

void apply_emboss(GtkWidget *widget, gpointer data) {

    cv::Mat kernel2D = generate_southemboss_kernel(kernel_size);

    apply_filter([kernel2D](const cv::Mat &img, int x, int y) -> cv::Vec3b {

        cv::Vec3d sum(0, 0, 0); 
        for (int y_counter = -kernel_size / 2; y_counter <= kernel_size / 2; y_counter++) {
            for (int x_counter = -kernel_size / 2; x_counter <= kernel_size / 2; x_counter++) {
                double weight = kernel2D.at<double>(y_counter + kernel_size / 2, x_counter + kernel_size / 2);
                sum += weight * static_cast<cv::Vec3d>(img.at<cv::Vec3b>(std::min(std::max(y + y_counter, 0), img.rows - 1),
                                                                         std::min(std::max(x + x_counter, 0), img.cols - 1)));
            }
        }

        return cv::Vec3b(
            cv::saturate_cast<uchar>(sum[0]),
            cv::saturate_cast<uchar>(sum[1]),
            cv::saturate_cast<uchar>(sum[2])
        );
    });

}

void update_image_display(GdkPixbuf *pixbuf) {
    if (!image_area || !pixbuf) return;

    gtk_image_set_from_pixbuf(GTK_IMAGE(image_area), pixbuf);
    gtk_widget_queue_draw(image_area);
}

void restore_original(GtkWidget *widget, gpointer data) {
    if (original_image.empty()) {
        std::cerr << "Error: No original image stored!" << std::endl;
        return;
    }

    image = original_image.clone();

    GdkPixbuf *pixbuf = gdk_pixbuf_new_from_data(
        image.data, GDK_COLORSPACE_RGB, FALSE, 8,
        image.cols, image.rows, image.step, NULL, NULL
    );

    gtk_image_set_from_pixbuf(GTK_IMAGE(image_area), pixbuf);
    gtk_widget_queue_draw(image_area);
}

void save_image(GtkWidget *widget, gpointer data) {

    if (image.empty()) {

        std::cerr << "Error: No image to save!" << std::endl;

        return;

    }



    GtkWidget *dialog = gtk_file_chooser_dialog_new("Save Image",

        GTK_WINDOW(data),

        GTK_FILE_CHOOSER_ACTION_SAVE,

        "_Cancel", GTK_RESPONSE_CANCEL,

        "_Save", GTK_RESPONSE_ACCEPT,

        NULL);



    if (gtk_dialog_run(GTK_DIALOG(dialog)) == GTK_RESPONSE_ACCEPT) {

        char *filename = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dialog));

        std::string save_path(filename);

        if (save_path.find('.') == std::string::npos) {

            save_path += ".png";

        }



        cv::Mat save_image;

        cv::cvtColor(image, save_image, cv::COLOR_RGB2BGR);



        if (!cv::imwrite(save_path, save_image)) {

            std::cerr << "Error: Failed to save the image!" << std::endl;

        } else {

            std::cout << "Image saved to: " << save_path << std::endl;

        }



        g_free(filename);

    }



    gtk_widget_destroy(dialog);

}

void load_image(GtkWidget *widget, gpointer data) {
    GtkWidget *dialog = gtk_file_chooser_dialog_new(
        "Open Image", GTK_WINDOW(data),
        GTK_FILE_CHOOSER_ACTION_OPEN,
        "_Cancel", GTK_RESPONSE_CANCEL,
        "_Open", GTK_RESPONSE_ACCEPT,
        NULL
    );

    if (gtk_dialog_run(GTK_DIALOG(dialog)) == GTK_RESPONSE_ACCEPT) {
        char *filename = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dialog));

        image = cv::imread(filename, cv::IMREAD_COLOR);

        if (!image.empty()) {
            cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

            original_image = image.clone();

            GdkPixbuf *pixbuf = gdk_pixbuf_new_from_data(
                image.data, GDK_COLORSPACE_RGB, FALSE, 8,
                image.cols, image.rows, image.step, NULL, NULL
            );

            update_image_display(pixbuf);
        } else {
            std::cerr << "Error: Failed to load image!" << std::endl;
        }

        g_free(filename);
    }

    gtk_widget_destroy(dialog);
}

void apply_graph_filter() {
    if (image.empty() || points.size() < 2) return;

    std::sort(points.begin(), points.end(), [](const GdkPoint &a, const GdkPoint &b) {
        return a.x < b.x;
    });

    apply_filter([](const cv::Mat &img, int x, int y) -> cv::Vec3b {
        
        cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);

        for (int channel = 0; channel < 3; channel++){
            int channel_value = static_cast<int>(pixel[channel]);
            for (size_t j = 0; j < points.size() - 1; j++) {
                if (channel_value >= points[j].x && channel_value <= points[j + 1].x) {
                    double x1 = points[j].x;
                    double y1 = points[j].y;
                    double x2 = points[j + 1].x;
                    double y2 = points[j + 1].y;


                    double t = (channel_value - x1) / (x2 - x1);
                    
                    pixel[channel] = cv::saturate_cast<uchar>(255 - (y1 + t * (y2 - y1)));
                    break;
                }
            }
        }
        return pixel;
    });

    GdkPixbuf *pixbuf = gdk_pixbuf_new_from_data(
        image.data, GDK_COLORSPACE_RGB, FALSE, 8,
        image.cols, image.rows, image.step, NULL, NULL
    );
    
    update_image_display(pixbuf);
}

void reset_filter() {

    points.clear();
    points.push_back({0, 255});
    points.push_back({255, 0});

    if (drawing_area) {
        gtk_widget_queue_draw(drawing_area);
    } else {
        std::cerr << "Error: drawing_area is NULL!\n";
    }
}

static gboolean on_mouse_click(GtkWidget *widget, GdkEventButton *event, gpointer data) {

    GdkPoint clicked_point = {static_cast<int>(event->x), static_cast<int>(event->y)};

    auto closest_it = std::min_element(points.begin(), points.end(), 
        [&clicked_point](const GdkPoint &a, const GdkPoint &b) {
            int dist_a = std::pow(a.x - clicked_point.x, 2) + std::pow(a.y - clicked_point.y, 2);
            int dist_b = std::pow(b.x - clicked_point.x, 2) + std::pow(b.y - clicked_point.y, 2);
            return dist_a < dist_b;
        });

    if (event->button == 1) {  

        if (closest_it != points.end() && std::abs(closest_it->x - clicked_point.x) < threshold) {

            closest_it->y = clicked_point.y;

        } else {
     
            auto it = std::lower_bound(points.begin(), points.end(), clicked_point,
                [](const GdkPoint &a, const GdkPoint &b) { return a.x < b.x; });
            points.insert(it, clicked_point);
        }

        gtk_widget_queue_draw(widget);
    }
    else if (event->button == 3) {
        if (!points.empty()) {
            if (closest_it != points.end() && closest_it->x != 0 && closest_it->x != 255) {
                points.erase(closest_it);
                gtk_widget_queue_draw(widget);
            }
        }
    }
    return TRUE;
}

static gboolean draw_function_graph(GtkWidget *widget, cairo_t *cr, gpointer data) {
    cairo_set_source_rgb(cr, 1, 1, 1); // White background
    cairo_paint(cr);

    // Draw Grid
    cairo_set_source_rgb(cr, 0.8, 0.8, 0.8);
    for (int i = 0; i <= 255; i += 64) {
        cairo_move_to(cr, i, 0);
        cairo_line_to(cr, i, 255);
        cairo_move_to(cr, 0, i);
        cairo_line_to(cr, 255, i);
    }
    cairo_stroke(cr);

    cairo_set_source_rgb(cr, 0, 1, 0); // Green Line
    cairo_set_line_width(cr, 2);
    for (size_t i = 0; i < points.size() - 1; i++) {
        cairo_move_to(cr, points[i].x, points[i].y);
        cairo_line_to(cr, points[i + 1].x, points[i + 1].y);
    }
    cairo_stroke(cr);

    cairo_set_source_rgb(cr, 1, 0, 0); // Red Points
    for (const auto &p : points) {
        cairo_arc(cr, p.x, p.y, 3, 0, 2 * M_PI);
        cairo_fill(cr);
    }

    return FALSE;
}

void save_filter(GtkWidget *widget, gpointer data) {
    GtkWidget *dialog = gtk_file_chooser_dialog_new(
        "Save Filter", GTK_WINDOW(data),
        GTK_FILE_CHOOSER_ACTION_SAVE,
        "_Cancel", GTK_RESPONSE_CANCEL,
        "_Save", GTK_RESPONSE_ACCEPT,
        NULL
    );

    gtk_file_chooser_set_do_overwrite_confirmation(GTK_FILE_CHOOSER(dialog), TRUE);

    if (gtk_dialog_run(GTK_DIALOG(dialog)) == GTK_RESPONSE_ACCEPT) {
        char *filename = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dialog));

        std::ofstream file(filename);
        if (file) {
            for (const auto &point : points) {
                file << point.x << " " << point.y << "\n";
            }
            file.close();
            std::cout << "Filter saved to " << filename << std::endl;
        } else {
            std::cerr << "Error: Could not open file for saving: " << filename << std::endl;
        }

        g_free(filename);
    }

    gtk_widget_destroy(dialog);
    
}

void load_filter(GtkWidget *widget, gpointer data) {
    GtkWidget *dialog = gtk_file_chooser_dialog_new(
        "Load Filter", GTK_WINDOW(data),
        GTK_FILE_CHOOSER_ACTION_OPEN,
        "_Cancel", GTK_RESPONSE_CANCEL,
        "_Open", GTK_RESPONSE_ACCEPT,
        NULL
    );

    if (gtk_dialog_run(GTK_DIALOG(dialog)) == GTK_RESPONSE_ACCEPT) {
        char *filename = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dialog));

        std::ifstream file(filename);
        if (file) {
            points.clear();
            GdkPoint point;
            while (file >> point.x >> point.y) {
                points.push_back(point);
            }
            file.close();
            std::cout << "Filter loaded from " << filename << std::endl;
        } else {
            std::cerr << "Error: Could not open file for loading: " << filename << std::endl;
        }

        g_free(filename);
    }

    gtk_widget_destroy(dialog);
    gtk_widget_queue_draw(widget);
}

void load_brightness_less(GtkWidget *widget, gpointer data){

    points.clear();
    GdkPoint point1, point2, point3;

    point1.x = 0; point1.y = 255;
    point2.x = 2*dbrightness; point2.y = 255;
    point3.x = 255; point3.y = 2*dbrightness;

    points.push_back(point1);
    points.push_back(point2);
    points.push_back(point3);


    gtk_widget_queue_draw(drawing_area); 
    gtk_widget_queue_resize(drawing_area);


}

void load_brightness_more(GtkWidget *widget, gpointer data){

    points.clear();
    GdkPoint point1, point2, point3;

    point1.x = 0; point1.y = 255 - 2*dbrightness;
    point2.x = 255 - 2*dbrightness; point2.y = 0;
    point3.x = 255; point3.y = 0;

    points.push_back(point1);
    points.push_back(point2);
    points.push_back(point3);


    gtk_widget_queue_draw(drawing_area); 
    gtk_widget_queue_resize(drawing_area);


}

void load_contrast_filter(GtkWidget *widget, gpointer data){

    points.clear();
    GdkPoint point1, point2, point3, point4;

    point1.x = 0; point1.y = 255;
    point2.x = (contrast_center *  contrast_coeff - contrast_center) / contrast_coeff; point2.y = 255;
    point3.x = (255 + contrast_center *  contrast_coeff - contrast_center) / contrast_coeff; point3.y = 0;
    point4.x = 255; point4.y = 0;

    points.push_back(point1);
    points.push_back(point2);
    points.push_back(point3);
    points.push_back(point4);

    gtk_widget_queue_draw(drawing_area); 
    gtk_widget_queue_resize(drawing_area);


}

void load_invertion_filter(GtkWidget *widget, gpointer data){

    points.clear();
    GdkPoint point1, point2, point3, point4;

    point1.x = 0; point1.y = 0;
    point2.x = 255; point2.y = 255;

    points.push_back(point1);
    points.push_back(point2);

    gtk_widget_queue_draw(drawing_area); 
    gtk_widget_queue_resize(drawing_area);


}

GtkWidget* create_menu_bar(GtkWidget *window) {

    // Setting up the menu bar
    GtkWidget *menu_bar = gtk_menu_bar_new();

    GtkWidget *file_menu = gtk_menu_new();
    GtkWidget *file_menu_item = gtk_menu_item_new_with_label("File");
    gtk_menu_item_set_submenu(GTK_MENU_ITEM(file_menu_item), file_menu);

    GtkWidget *load_item = gtk_menu_item_new_with_label("Load Image");
    GtkWidget *save_item = gtk_menu_item_new_with_label("Save Image");
    GtkWidget *load_filter_ = gtk_menu_item_new_with_label("Load Filter");
    GtkWidget *save_filter_ = gtk_menu_item_new_with_label("Save Filter");
    GtkWidget *exit_item = gtk_menu_item_new_with_label("Exit");

    gtk_menu_shell_append(GTK_MENU_SHELL(file_menu), load_item);
    gtk_menu_shell_append(GTK_MENU_SHELL(file_menu), save_item);
    gtk_menu_shell_append(GTK_MENU_SHELL(file_menu), load_filter_);
    gtk_menu_shell_append(GTK_MENU_SHELL(file_menu), save_filter_);
    gtk_menu_shell_append(GTK_MENU_SHELL(file_menu), exit_item);

    g_signal_connect(load_item, "activate", G_CALLBACK(load_image), window);
    g_signal_connect(save_item, "activate", G_CALLBACK(save_image), window);
    g_signal_connect(load_filter_, "activate", G_CALLBACK(load_filter), window);
    g_signal_connect(save_filter_, "activate", G_CALLBACK(save_filter), window);
    g_signal_connect(exit_item, "activate", G_CALLBACK(gtk_main_quit), NULL);

    gtk_menu_shell_append(GTK_MENU_SHELL(menu_bar), file_menu_item);

    //Pixel filters
{

    GtkWidget *filter_menu = gtk_menu_new();
    GtkWidget *filter_menu_item = gtk_menu_item_new_with_label("Filters");
    gtk_menu_item_set_submenu(GTK_MENU_ITEM(filter_menu_item), filter_menu);

    GtkWidget *invert_item = gtk_menu_item_new_with_label("Invert Colors");
    GtkWidget *add_brightness = gtk_menu_item_new_with_label("Add brightness");
    GtkWidget *remove_brightness = gtk_menu_item_new_with_label("Remove brightness");
    GtkWidget *contrast_option = gtk_menu_item_new_with_label("Contrast");
    GtkWidget *add_gamma = gtk_menu_item_new_with_label("Gamma bright");
    GtkWidget *remove_gamma = gtk_menu_item_new_with_label("Gamma dark");
    GtkWidget *restore_item = gtk_menu_item_new_with_label("Restore Original");



    gtk_menu_shell_append(GTK_MENU_SHELL(filter_menu), invert_item);
    gtk_menu_shell_append(GTK_MENU_SHELL(filter_menu), add_brightness);
    gtk_menu_shell_append(GTK_MENU_SHELL(filter_menu), remove_brightness);
    gtk_menu_shell_append(GTK_MENU_SHELL(filter_menu), contrast_option);
    gtk_menu_shell_append(GTK_MENU_SHELL(filter_menu), add_gamma);
    gtk_menu_shell_append(GTK_MENU_SHELL(filter_menu), remove_gamma);
    gtk_menu_shell_append(GTK_MENU_SHELL(filter_menu), restore_item);


    g_signal_connect(invert_item, "activate", G_CALLBACK(apply_inversion), NULL);
    g_signal_connect(add_brightness, "activate", G_CALLBACK(apply_more_brightness),NULL);
    g_signal_connect(remove_brightness, "activate", G_CALLBACK(apply_less_brightness),NULL);
    g_signal_connect(contrast_option, "activate", G_CALLBACK(apply_contrast),NULL);
    g_signal_connect(add_gamma, "activate", G_CALLBACK(apply_gamma_bright),NULL);
    g_signal_connect(remove_gamma, "activate", G_CALLBACK(apply_gamma_dark),NULL);
    g_signal_connect(restore_item, "activate", G_CALLBACK(restore_original), NULL);


    gtk_menu_shell_append(GTK_MENU_SHELL(menu_bar), filter_menu_item);

}

    // Kernel filters
{

    GtkWidget *kfilter_menu = gtk_menu_new();
    GtkWidget *kfilter_menu_item = gtk_menu_item_new_with_label("Kernel Filters");
    gtk_menu_item_set_submenu(GTK_MENU_ITEM(kfilter_menu_item), kfilter_menu);

    GtkWidget *blur_option = gtk_menu_item_new_with_label("Blur");
    GtkWidget *gaussblur_option = gtk_menu_item_new_with_label("Gauss Smoothing");
    GtkWidget *sharpen_option = gtk_menu_item_new_with_label("Sharpening");
    GtkWidget *edges_option = gtk_menu_item_new_with_label("Edge Detection");
    GtkWidget *emboss_option = gtk_menu_item_new_with_label("Emboss Filter");

    gtk_menu_shell_append(GTK_MENU_SHELL(kfilter_menu), blur_option);
    gtk_menu_shell_append(GTK_MENU_SHELL(kfilter_menu), gaussblur_option);
    gtk_menu_shell_append(GTK_MENU_SHELL(kfilter_menu), sharpen_option);
    gtk_menu_shell_append(GTK_MENU_SHELL(kfilter_menu), edges_option);
    gtk_menu_shell_append(GTK_MENU_SHELL(kfilter_menu), emboss_option);

    g_signal_connect(blur_option, "activate", G_CALLBACK(apply_blur), NULL);
    g_signal_connect(gaussblur_option, "activate", G_CALLBACK(apply_gaussblur), NULL);
    g_signal_connect(sharpen_option, "activate", G_CALLBACK(apply_sharpen), NULL);
    g_signal_connect(edges_option, "activate", G_CALLBACK(apply_edges), NULL);
    g_signal_connect(emboss_option, "activate", G_CALLBACK(apply_emboss), NULL);

    gtk_menu_shell_append(GTK_MENU_SHELL(menu_bar), kfilter_menu_item);

}

    // Functional filter
{
    GtkWidget *ffilter_menu = gtk_menu_new();
    GtkWidget *ffilter_menu_item = gtk_menu_item_new_with_label("Functional Filter");
    gtk_menu_item_set_submenu(GTK_MENU_ITEM(ffilter_menu_item), ffilter_menu);

    GtkWidget *custom_filter = gtk_menu_item_new_with_label("Apply");
    GtkWidget *reset_filter_ = gtk_menu_item_new_with_label("Reset Filter");
    GtkWidget *less_brightness_load = gtk_menu_item_new_with_label("Load -Brightness");
    GtkWidget *more_brightness_load = gtk_menu_item_new_with_label("Load +Brightness");
    GtkWidget *contrast_load = gtk_menu_item_new_with_label("Load Contrast");
    GtkWidget *invertion_load = gtk_menu_item_new_with_label("Load Invertion");

    gtk_menu_shell_append(GTK_MENU_SHELL(ffilter_menu), custom_filter);
    gtk_menu_shell_append(GTK_MENU_SHELL(ffilter_menu), reset_filter_);
    gtk_menu_shell_append(GTK_MENU_SHELL(ffilter_menu), less_brightness_load);
    gtk_menu_shell_append(GTK_MENU_SHELL(ffilter_menu), more_brightness_load);
    gtk_menu_shell_append(GTK_MENU_SHELL(ffilter_menu), contrast_load);
    gtk_menu_shell_append(GTK_MENU_SHELL(ffilter_menu), invertion_load);

    g_signal_connect(custom_filter, "activate", G_CALLBACK(apply_graph_filter), NULL);
    g_signal_connect(reset_filter_, "activate", G_CALLBACK(reset_filter), NULL);
    g_signal_connect(less_brightness_load, "activate", G_CALLBACK(load_brightness_less), NULL);
    g_signal_connect(more_brightness_load, "activate", G_CALLBACK(load_brightness_more), NULL);
    g_signal_connect(contrast_load, "activate", G_CALLBACK(load_contrast_filter), NULL);
    g_signal_connect(invertion_load, "activate", G_CALLBACK(load_invertion_filter), NULL);

    gtk_menu_shell_append(GTK_MENU_SHELL(menu_bar), ffilter_menu_item);
}

    // Pixelize
{
    GtkWidget *pixelize_menu = gtk_menu_new();
    GtkWidget *pixelize_menu_item = gtk_menu_item_new_with_label("Pixelize");
    gtk_menu_item_set_submenu(GTK_MENU_ITEM(pixelize_menu_item), pixelize_menu);


    GtkWidget *apply_pixelize = gtk_menu_item_new_with_label("Apply");

    gtk_menu_shell_append(GTK_MENU_SHELL(pixelize_menu), apply_pixelize);

    g_signal_connect(apply_pixelize, "activate", G_CALLBACK(pixelize),NULL);

    gtk_menu_shell_append(GTK_MENU_SHELL(menu_bar), pixelize_menu_item);

}
    //Greyscale
{
    GtkWidget *grey_menu = gtk_menu_new();
    GtkWidget *grey_menu_item = gtk_menu_item_new_with_label("Grey");
    gtk_menu_item_set_submenu(GTK_MENU_ITEM(grey_menu_item), grey_menu);

    GtkWidget *apply_greying = gtk_menu_item_new_with_label("Apply greying");
    GtkWidget *apply_dith = gtk_menu_item_new_with_label("Apply Random Dithering");

    gtk_menu_shell_append(GTK_MENU_SHELL(grey_menu), apply_greying);
    gtk_menu_shell_append(GTK_MENU_SHELL(grey_menu), apply_dith);

    g_signal_connect(apply_greying, "activate", G_CALLBACK(apply_greyscale),NULL);
    g_signal_connect(apply_dith, "activate", G_CALLBACK(apply_random_dithering),NULL);

    gtk_menu_shell_append(GTK_MENU_SHELL(menu_bar), grey_menu_item);

}

    return menu_bar;

}

int main(int argc, char *argv[]) {
    gtk_init(&argc, &argv);

    GtkWidget *window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_title(GTK_WINDOW(window), "Function Graph Editor");
    gtk_window_set_default_size(GTK_WINDOW(window), 1900, 1000);

    GtkWidget *main_vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    gtk_container_add(GTK_CONTAINER(window), main_vbox);

    GtkWidget *menu_bar = create_menu_bar(window);
    gtk_box_pack_start(GTK_BOX(main_vbox), menu_bar, FALSE, FALSE, 0);

    GtkWidget *paned = gtk_paned_new(GTK_ORIENTATION_HORIZONTAL);
    gtk_box_pack_start(GTK_BOX(main_vbox), paned, TRUE, TRUE, 0);

    scrolled_window = gtk_scrolled_window_new(NULL, NULL);
    image_area = gtk_image_new();
    gtk_container_add(GTK_CONTAINER(scrolled_window), image_area);
    gtk_paned_pack1(GTK_PANED(paned), scrolled_window, TRUE, FALSE);

    GtkWidget *graph_container = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
    
    GtkWidget *slider_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);

    drawing_area = gtk_drawing_area_new();
    gtk_widget_set_size_request(drawing_area, 256, 256);

    gtk_widget_set_margin_start(drawing_area, 20);
    gtk_widget_set_margin_end(drawing_area, 20);
    gtk_widget_set_margin_top(drawing_area, 20);
    gtk_widget_set_margin_bottom(drawing_area, 20);

    gtk_widget_set_halign(drawing_area, GTK_ALIGN_CENTER);
    gtk_widget_set_valign(drawing_area, GTK_ALIGN_CENTER);

    GtkWidget *slider = create_labeled_shades_slider();

    gtk_box_pack_start(GTK_BOX(slider_box), drawing_area, TRUE, TRUE, 10);
    gtk_box_pack_start(GTK_BOX(slider_box), slider, FALSE, FALSE, 5);

    gtk_box_pack_start(GTK_BOX(graph_container), slider_box, TRUE, TRUE, 10);
    gtk_paned_pack2(GTK_PANED(paned), graph_container, TRUE, FALSE);

    gtk_paned_set_position(GTK_PANED(paned), 1200);

    gtk_widget_add_events(drawing_area, GDK_BUTTON_PRESS_MASK | GDK_BUTTON_RELEASE_MASK);
    g_signal_connect(drawing_area, "draw", G_CALLBACK(draw_function_graph), NULL);
    g_signal_connect(drawing_area, "button-press-event", G_CALLBACK(on_mouse_click), drawing_area); 

    g_signal_connect(window, "destroy", G_CALLBACK(gtk_main_quit), NULL);
    gtk_widget_show_all(window);
    gtk_main();

    return 0;
}


