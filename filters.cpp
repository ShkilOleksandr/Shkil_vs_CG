#include <gtk/gtk.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <functional>
#include <math.h>

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

void load_image(GtkWidget *widget, gpointer data) {
    GtkWidget *dialog = gtk_file_chooser_dialog_new("Open Image",
        GTK_WINDOW(data),
        GTK_FILE_CHOOSER_ACTION_OPEN,
        "_Cancel", GTK_RESPONSE_CANCEL,
        "_Open", GTK_RESPONSE_ACCEPT,
        NULL);

    if (gtk_dialog_run(GTK_DIALOG(dialog)) == GTK_RESPONSE_ACCEPT) {
        char *filename = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dialog));
        original_image = cv::imread(filename, cv::IMREAD_COLOR);
        if (original_image.empty()) {
            std::cerr << "Error loading image!" << std::endl;
            return;
        }

        cv::cvtColor(original_image, image, cv::COLOR_BGR2RGB);
        g_free(filename);

        GdkPixbuf *pixbuf = gdk_pixbuf_new_from_data(
            image.data, GDK_COLORSPACE_RGB, FALSE, 8,
            image.cols, image.rows, image.step, NULL, NULL
        );

        gtk_image_set_from_pixbuf(GTK_IMAGE(image_area), pixbuf);
        gtk_widget_set_size_request(image_area, image.cols, image.rows);
    }
    gtk_widget_destroy(dialog);
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

void restore_original(GtkWidget *widget, gpointer data) {
    if (original_image.empty()) return;

    cv::cvtColor(original_image, image, cv::COLOR_BGR2RGB);

    GdkPixbuf *pixbuf = gdk_pixbuf_new_from_data(
        image.data, GDK_COLORSPACE_RGB, FALSE, 8,
        image.cols, image.rows, image.step, NULL, NULL
    );

    gtk_image_set_from_pixbuf(GTK_IMAGE(image_area), pixbuf);
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

GtkWidget* create_menu_bar(GtkWidget *window) {

    // Setting up the menu bar

    GtkWidget *menu_bar = gtk_menu_bar_new();

    GtkWidget *file_menu = gtk_menu_new();
    GtkWidget *file_menu_item = gtk_menu_item_new_with_label("File");
    gtk_menu_item_set_submenu(GTK_MENU_ITEM(file_menu_item), file_menu);

    GtkWidget *load_item = gtk_menu_item_new_with_label("Load Image");
    GtkWidget *save_item = gtk_menu_item_new_with_label("Save Image");
    GtkWidget *exit_item = gtk_menu_item_new_with_label("Exit");

    gtk_menu_shell_append(GTK_MENU_SHELL(file_menu), load_item);
    gtk_menu_shell_append(GTK_MENU_SHELL(file_menu), save_item);
    gtk_menu_shell_append(GTK_MENU_SHELL(file_menu), exit_item);

    g_signal_connect(load_item, "activate", G_CALLBACK(load_image), window);
    g_signal_connect(save_item, "activate", G_CALLBACK(save_image), window);
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
    return menu_bar;
}

int main(int argc, char *argv[]) {
    gtk_init(&argc, &argv);

    GtkWidget *window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_title(GTK_WINDOW(window), "Image Filtering App");
    gtk_window_set_default_size(GTK_WINDOW(window), 800, 600);
    gtk_container_set_border_width(GTK_CONTAINER(window), 10);

    GtkWidget *vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    gtk_container_add(GTK_CONTAINER(window), vbox);

    GtkWidget *menu_bar = create_menu_bar(window);
    gtk_box_pack_start(GTK_BOX(vbox), menu_bar, FALSE, FALSE, 0);

    scrolled_window = gtk_scrolled_window_new(NULL, NULL);
    gtk_widget_set_vexpand(scrolled_window, TRUE);
    gtk_widget_set_hexpand(scrolled_window, TRUE);
    gtk_box_pack_start(GTK_BOX(vbox), scrolled_window, TRUE, TRUE, 5);

    image_area = gtk_image_new();
    gtk_container_add(GTK_CONTAINER(scrolled_window), image_area);

    g_signal_connect(window, "destroy", G_CALLBACK(gtk_main_quit), NULL);
    gtk_widget_show_all(window);
    gtk_main();

    return 0;
}
