#include <gtk/gtk.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <functional>
#include <math.h>
#include <vector>
#include <fstream> 

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

// Custom Grid manipulation variables
int ckernel_size = 3;
std::vector<std::vector<GtkWidget*>> kernel_entries;
std::vector<std::vector<int>> kernel_values;
GtkWidget *size_spin; 
GtkWidget *divisor_entry;
GtkWidget *apply_button;
GtkWidget *auto_divisor_button;
GtkWidget *offset_entry;
GtkWidget *anchor_x_spin, *anchor_y_spin;

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

void apply_custom_filter(GtkWidget *widget, gpointer data) {
    if (image.empty()) return;

    kernel_values = std::vector<std::vector<int>>(ckernel_size, std::vector<int>(ckernel_size, 0));

    int offset = (9 - ckernel_size) / 2;

    for (int y = 0; y < ckernel_size; y++) {
        for (int x = 0; x < ckernel_size; x++) {
            int entry_y = y + offset;
            int entry_x = x + offset;

            const char *text = gtk_entry_get_text(GTK_ENTRY(kernel_entries[entry_y][entry_x]));

            if (text && strlen(text) > 0) {
                try {
                    kernel_values[y][x] = std::stoi(text);
                } catch (...) {
                    kernel_values[y][x] = 0;
                }
            } else {
                kernel_values[y][x] = 0;
            }
        }
    }

    const char *offset_text = gtk_entry_get_text(GTK_ENTRY(offset_entry));
    if (offset_text && strlen(offset_text) > 0) {
        try {
            offset = std::stoi(offset_text);
        } catch (...) {
            offset = 0;
        }
    }

    int divisor = 1;
    const char *divisor_text = gtk_entry_get_text(GTK_ENTRY(divisor_entry));
    if (divisor_text && strlen(divisor_text) > 0) {
        try {
            divisor = std::stoi(divisor_text);
            if (divisor == 0) divisor = 1;
        } catch (...) {
            divisor = 1;
        }
    }

    int anchor_x = gtk_spin_button_get_value_as_int(GTK_SPIN_BUTTON(anchor_x_spin));
    int anchor_y = gtk_spin_button_get_value_as_int(GTK_SPIN_BUTTON(anchor_y_spin));

    apply_filter([divisor, offset, anchor_x, anchor_y](const cv::Mat &img, int x, int y) -> cv::Vec3b {
        cv::Vec3d sum(0, 0, 0); 

        for (int j = 0; j < ckernel_size; j++) {
            for (int i = 0; i < ckernel_size; i++) {
                int new_y = std::clamp(y + j - anchor_y, 0, img.rows - 1);
                int new_x = std::clamp(x + i - anchor_x, 0, img.cols - 1);

                int weight = kernel_values[j][i];
                sum += weight * static_cast<cv::Vec3d>(img.at<cv::Vec3b>(new_y, new_x));
            }
        }


        sum /= divisor;
        sum += cv::Vec3d(offset, offset, offset);

        return cv::Vec3b(
            cv::saturate_cast<uchar>(sum[0]),
            cv::saturate_cast<uchar>(sum[1]),
            cv::saturate_cast<uchar>(sum[2])
        );
    });
}

void update_ckernel_size(GtkWidget *widget, gpointer data) {
    ckernel_size = gtk_spin_button_get_value_as_int(GTK_SPIN_BUTTON(widget));

    for (int y = 0; y < 9; y++) {
        for (int x = 0; x < 9; x++) {
            bool is_active = (abs(y - 4) < ckernel_size / 2 + 1) && (abs(x - 4) < ckernel_size / 2 + 1);
            gtk_widget_set_sensitive(kernel_entries[y][x], is_active);
        }
    }
}

void auto_compute_divisor(GtkWidget *widget, gpointer data) {
    int sum = 0;

    for (int y = 0; y < 9; y++) {
        for (int x = 0; x < 9; x++) {
            
            bool is_active = (abs(y - 4) < ckernel_size / 2 + 1) && (abs(x - 4) < ckernel_size / 2 + 1);

            if(is_active){
                const char *text = gtk_entry_get_text(GTK_ENTRY(kernel_entries[y][x]));
                if (text && strlen(text) > 0) {
                    try {
                        sum += std::stoi(text);
                    } catch (...) {
                        sum += 0;
                    }
                }
            }
        }
    }
    if (sum == 0) sum = 1;

    std::string divisor_text = std::to_string(sum);

    gtk_entry_set_text(GTK_ENTRY(divisor_entry), divisor_text.c_str());
}

GtkWidget *create_kernel_editor() {
    GtkWidget *main_hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
    GtkWidget *kernel_frame, *grid, *control_vbox;
    
    kernel_frame = gtk_frame_new("Convolution Kernel");
    grid = gtk_grid_new();
    gtk_grid_set_column_spacing(GTK_GRID(grid), 5);
    gtk_grid_set_row_spacing(GTK_GRID(grid), 5);
    gtk_grid_set_column_homogeneous(GTK_GRID(grid), FALSE);
    gtk_container_add(GTK_CONTAINER(kernel_frame), grid);
    
    kernel_entries.resize(9, std::vector<GtkWidget*>(9));
    for (int y = 0; y < 9; y++) {
        for (int x = 0; x < 9; x++) {
            kernel_entries[y][x] = gtk_entry_new();
            gtk_entry_set_max_length(GTK_ENTRY(kernel_entries[y][x]), 2);
            gtk_entry_set_width_chars(GTK_ENTRY(kernel_entries[y][x]), 2);
            gtk_widget_set_hexpand(kernel_entries[y][x], FALSE);
            gtk_widget_set_vexpand(kernel_entries[y][x], FALSE);
            gtk_widget_set_size_request(kernel_entries[y][x], 25, 25);
            gtk_grid_attach(GTK_GRID(grid), kernel_entries[y][x], x, y, 1, 1);
        }
    }

    control_vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    
    GtkWidget* size_label = gtk_label_new("Kernel Size:");
    size_spin = gtk_spin_button_new_with_range(1, 9, 2);
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(size_spin), ckernel_size);
    g_signal_connect(size_spin, "value-changed", G_CALLBACK(update_ckernel_size), NULL);

    GtkWidget *size_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    gtk_box_pack_start(GTK_BOX(size_box), size_label, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(size_box), size_spin, FALSE, FALSE, 0);

    gtk_box_pack_start(GTK_BOX(control_vbox), size_box, FALSE, FALSE, 0);

    GtkWidget* divisor_label = gtk_label_new("Divisor:");
    divisor_entry = gtk_entry_new();
    gtk_entry_set_text(GTK_ENTRY(divisor_entry), "1");
    gtk_entry_set_width_chars(GTK_ENTRY(divisor_entry), 3);
    
    GtkWidget *divisor_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    gtk_box_pack_start(GTK_BOX(divisor_box), divisor_label, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(divisor_box), divisor_entry, FALSE, FALSE, 0);

    gtk_box_pack_start(GTK_BOX(control_vbox), divisor_box, FALSE, FALSE, 0);

    auto_divisor_button = gtk_button_new_with_label("Auto Divisor");
    g_signal_connect(auto_divisor_button, "clicked", G_CALLBACK(auto_compute_divisor), NULL);
    gtk_box_pack_start(GTK_BOX(control_vbox), auto_divisor_button, FALSE, FALSE, 0);

    GtkWidget *offset_label = gtk_label_new("Offset:");
    offset_entry = gtk_entry_new();
    gtk_entry_set_text(GTK_ENTRY(offset_entry), "0");
    gtk_entry_set_width_chars(GTK_ENTRY(offset_entry), 3);

    GtkWidget *offset_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    gtk_box_pack_start(GTK_BOX(offset_box), offset_label, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(offset_box), offset_entry, FALSE, FALSE, 0);

    gtk_box_pack_start(GTK_BOX(control_vbox), offset_box, FALSE, FALSE, 0);

    GtkWidget *anchor_x_label = gtk_label_new("Anchor X:");
    anchor_x_spin = gtk_spin_button_new_with_range(0, ckernel_size - 1, 1);
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(anchor_x_spin), ckernel_size / 2);

    GtkWidget *anchor_y_label = gtk_label_new("Anchor Y:");
    anchor_y_spin = gtk_spin_button_new_with_range(0, ckernel_size - 1, 1);
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(anchor_y_spin), ckernel_size / 2);

    GtkWidget *anchor_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    gtk_box_pack_start(GTK_BOX(anchor_box), anchor_x_label, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(anchor_box), anchor_x_spin, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(anchor_box), anchor_y_label, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(anchor_box), anchor_y_spin, FALSE, FALSE, 0);

    gtk_box_pack_start(GTK_BOX(control_vbox), anchor_box, FALSE, FALSE, 0);

    apply_button = gtk_button_new_with_label("Apply Filter");
    g_signal_connect(apply_button, "clicked", G_CALLBACK(apply_custom_filter), NULL);
    gtk_box_pack_start(GTK_BOX(control_vbox), apply_button, FALSE, FALSE, 0);

    gtk_box_pack_start(GTK_BOX(main_hbox), kernel_frame, FALSE, FALSE, 5);
    gtk_box_pack_start(GTK_BOX(main_hbox), control_vbox, FALSE, FALSE, 5);

    update_ckernel_size(size_spin, NULL);

    return main_hbox;
}

void save_filter(const char* filename) {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Error saving filter!" << std::endl;
        return;
    }

    file << ckernel_size << std::endl;

    int start = (9 - ckernel_size) / 2; 

    for (int y = 0; y < ckernel_size; y++) {
        for (int x = 0; x < ckernel_size; x++) {
            const char *text = gtk_entry_get_text(GTK_ENTRY(kernel_entries[start + y][start + x]));
            file << (text && strlen(text) > 0 ? text : "0") << " ";
        }
        file << std::endl;
    }

    file << gtk_entry_get_text(GTK_ENTRY(divisor_entry)) << std::endl;
    file << gtk_entry_get_text(GTK_ENTRY(offset_entry)) << std::endl;
    file << gtk_spin_button_get_value_as_int(GTK_SPIN_BUTTON(anchor_x_spin)) << " ";
    file << gtk_spin_button_get_value_as_int(GTK_SPIN_BUTTON(anchor_y_spin)) << std::endl;

    file.close();
}

void load_filter(const char* filename) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Error loading filter!" << std::endl;
        return;
    }

    file >> ckernel_size;
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(size_spin), ckernel_size);
    update_ckernel_size(size_spin, NULL);

    int start = (9 - ckernel_size) / 2;

    for (int y = 0; y < 9; y++) {
        for (int x = 0; x < 9; x++) {
            gtk_entry_set_text(GTK_ENTRY(kernel_entries[y][x]), "");
        }
    }

    for (int y = 0; y < ckernel_size; y++) {
        for (int x = 0; x < ckernel_size; x++) {
            std::string value;
            file >> value;
            gtk_entry_set_text(GTK_ENTRY(kernel_entries[start + y][start + x]), value.c_str());
        }
    }

    std::string divisor, offset;
    int anchor_x, anchor_y;
    file >> divisor >> offset >> anchor_x >> anchor_y;

    gtk_entry_set_text(GTK_ENTRY(divisor_entry), divisor.c_str());
    gtk_entry_set_text(GTK_ENTRY(offset_entry), offset.c_str());
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(anchor_x_spin), anchor_x);
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(anchor_y_spin), anchor_y);

    file.close();
}

void on_load_filter(GtkWidget *widget, gpointer data) {
    GtkWidget *dialog = gtk_file_chooser_dialog_new(
        "Open Filter", GTK_WINDOW(data), GTK_FILE_CHOOSER_ACTION_OPEN,
        "_Cancel", GTK_RESPONSE_CANCEL,
        "_Open", GTK_RESPONSE_ACCEPT,
        NULL
    );

    if (gtk_dialog_run(GTK_DIALOG(dialog)) == GTK_RESPONSE_ACCEPT) {
        char *filename = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dialog));
        load_filter(filename);
        g_free(filename);
    }

    gtk_widget_destroy(dialog);
}

void on_save_filter(GtkWidget *widget, gpointer data) {
    GtkWidget *dialog = gtk_file_chooser_dialog_new(
        "Save Filter", GTK_WINDOW(data), GTK_FILE_CHOOSER_ACTION_SAVE,
        "_Cancel", GTK_RESPONSE_CANCEL,
        "_Save", GTK_RESPONSE_ACCEPT,
        NULL
    );

    gtk_file_chooser_set_do_overwrite_confirmation(GTK_FILE_CHOOSER(dialog), TRUE);

    if (gtk_dialog_run(GTK_DIALOG(dialog)) == GTK_RESPONSE_ACCEPT) {
        char *filename = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dialog));
        save_filter(filename);
        g_free(filename);
    }

    gtk_widget_destroy(dialog);
}

GtkWidget* create_menu_bar(GtkWidget *window) {

    GtkWidget *menu_bar = gtk_menu_bar_new();

    // File Menu

    GtkWidget *file_menu = gtk_menu_new();
    GtkWidget *file_menu_item = gtk_menu_item_new_with_label("File");
    gtk_menu_item_set_submenu(GTK_MENU_ITEM(file_menu_item), file_menu);

    GtkWidget *load_item = gtk_menu_item_new_with_label("Load Image");
    GtkWidget *save_item = gtk_menu_item_new_with_label("Save Image");
    GtkWidget *load_filter_item = gtk_menu_item_new_with_label("Load Filter"); 
    GtkWidget *save_filter_item = gtk_menu_item_new_with_label("Save Filter");
    GtkWidget *exit_item = gtk_menu_item_new_with_label("Exit");

    gtk_menu_shell_append(GTK_MENU_SHELL(file_menu), load_item);
    gtk_menu_shell_append(GTK_MENU_SHELL(file_menu), save_item);
    gtk_menu_shell_append(GTK_MENU_SHELL(file_menu), load_filter_item); 
    gtk_menu_shell_append(GTK_MENU_SHELL(file_menu), save_filter_item);
    gtk_menu_shell_append(GTK_MENU_SHELL(file_menu), exit_item);

    g_signal_connect(load_item, "activate", G_CALLBACK(load_image), window);
    g_signal_connect(save_item, "activate", G_CALLBACK(save_image), window);
    g_signal_connect(load_filter_item, "activate", G_CALLBACK(on_load_filter), window);
    g_signal_connect(save_filter_item, "activate", G_CALLBACK(on_save_filter), window);
    g_signal_connect(exit_item, "activate", G_CALLBACK(gtk_main_quit), NULL);

    gtk_menu_shell_append(GTK_MENU_SHELL(menu_bar), file_menu_item);

    // Pixel Filters Menu
    {
        GtkWidget *filter_menu = gtk_menu_new();
        GtkWidget *filter_menu_item = gtk_menu_item_new_with_label("Filters");
        gtk_menu_item_set_submenu(GTK_MENU_ITEM(filter_menu_item), filter_menu);

        GtkWidget *invert_item = gtk_menu_item_new_with_label("Invert Colors");
        GtkWidget *add_brightness = gtk_menu_item_new_with_label("Add Brightness");
        GtkWidget *remove_brightness = gtk_menu_item_new_with_label("Remove Brightness");
        GtkWidget *contrast_option = gtk_menu_item_new_with_label("Contrast");
        GtkWidget *add_gamma = gtk_menu_item_new_with_label("Gamma Bright");
        GtkWidget *remove_gamma = gtk_menu_item_new_with_label("Gamma Dark");
        GtkWidget *restore_item = gtk_menu_item_new_with_label("Restore Original");

        gtk_menu_shell_append(GTK_MENU_SHELL(filter_menu), invert_item);
        gtk_menu_shell_append(GTK_MENU_SHELL(filter_menu), add_brightness);
        gtk_menu_shell_append(GTK_MENU_SHELL(filter_menu), remove_brightness);
        gtk_menu_shell_append(GTK_MENU_SHELL(filter_menu), contrast_option);
        gtk_menu_shell_append(GTK_MENU_SHELL(filter_menu), add_gamma);
        gtk_menu_shell_append(GTK_MENU_SHELL(filter_menu), remove_gamma);
        gtk_menu_shell_append(GTK_MENU_SHELL(filter_menu), restore_item);

        g_signal_connect(invert_item, "activate", G_CALLBACK(apply_inversion), NULL);
        g_signal_connect(add_brightness, "activate", G_CALLBACK(apply_more_brightness), NULL);
        g_signal_connect(remove_brightness, "activate", G_CALLBACK(apply_less_brightness), NULL);
        g_signal_connect(contrast_option, "activate", G_CALLBACK(apply_contrast), NULL);
        g_signal_connect(add_gamma, "activate", G_CALLBACK(apply_gamma_bright), NULL);
        g_signal_connect(remove_gamma, "activate", G_CALLBACK(apply_gamma_dark), NULL);
        g_signal_connect(restore_item, "activate", G_CALLBACK(restore_original), NULL);

        gtk_menu_shell_append(GTK_MENU_SHELL(menu_bar), filter_menu_item);
    }

    // Kernel Filters Menu
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
    gtk_window_set_default_size(GTK_WINDOW(window), 1900, 1000);
    gtk_container_set_border_width(GTK_CONTAINER(window), 10);

    GtkWidget *vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    gtk_container_add(GTK_CONTAINER(window), vbox);

    GtkWidget *menu_bar = create_menu_bar(window);
    gtk_box_pack_start(GTK_BOX(vbox), menu_bar, FALSE, FALSE, 0);

    GtkWidget *paned = gtk_paned_new(GTK_ORIENTATION_HORIZONTAL);
    gtk_box_pack_start(GTK_BOX(vbox), paned, TRUE, TRUE, 0);

    GtkWidget *image_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    scrolled_window = gtk_scrolled_window_new(NULL, NULL);
    gtk_widget_set_vexpand(scrolled_window, TRUE);
    gtk_widget_set_hexpand(scrolled_window, TRUE);
    gtk_box_pack_start(GTK_BOX(image_box), scrolled_window, TRUE, TRUE, 5);

    image_area = gtk_image_new();
    gtk_container_add(GTK_CONTAINER(scrolled_window), image_area);

    gtk_paned_add1(GTK_PANED(paned), image_box);

    GtkWidget *kernel_editor = create_kernel_editor();
    gtk_paned_add2(GTK_PANED(paned), kernel_editor);

    gtk_paned_set_position(GTK_PANED(paned), 1000);

    g_signal_connect(window, "destroy", G_CALLBACK(gtk_main_quit), NULL);

    gtk_widget_show_all(window);
    gtk_main();

    return 0;
}



