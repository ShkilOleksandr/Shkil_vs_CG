#include <gtk/gtk.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

struct Line {
    cv::Point start, end;
    int thickness; // logical thickness
};

std::vector<Line> lines;
cv::Mat image;
GtkWidget* drawing_area;
cv::Point lineStart;
bool awaitingSecondClick = false;
double current_scale = 1.0; // updated during draw to scale thickness

enum class EditMode { None, MoveStart, MoveEnd, AdjustThickness };
EditMode editMode = EditMode::None;
int selectedLineIndex = -1;


void drawLineDDA(cv::Mat& img, cv::Point p1, cv::Point p2, const cv::Vec3b& color, int thickness = 1) {
    int dx = p2.x - p1.x;
    int dy = p2.y - p1.y;
    int steps = std::max(std::abs(dx), std::abs(dy));

    float x_inc = dx / static_cast<float>(steps);
    float y_inc = dy / static_cast<float>(steps);
    float x = p1.x, y = p1.y;

    int half = (thickness - 1) / 2;
    //half = 0;

    for (int i = 0; i <= steps; ++i) {
        int px = static_cast<int>(std::round(x));
        int py = static_cast<int>(std::round(y));

        for (int dy = -half; dy <= half; ++dy) {
            for (int dx = -half; dx <= half; ++dx) {
                int nx = px + dx;
                int ny = py + dy;

                if (nx >= 0 && nx < img.cols && ny >= 0 && ny < img.rows) {
                    img.at<cv::Vec3b>(ny, nx) = color;
                }
            }
        }

        x += x_inc;
        y += y_inc;
    }
}


cv::Point map_click_to_image(GtkWidget* widget, double click_x, double click_y) {
    int w = gtk_widget_get_allocated_width(widget);
    int h = gtk_widget_get_allocated_height(widget);
    double scale_x = static_cast<double>(w) / image.cols;
    double scale_y = static_cast<double>(h) / image.rows;
    double scale = std::min(scale_x, scale_y);

    int draw_w = static_cast<int>(image.cols * scale);
    int draw_h = static_cast<int>(image.rows * scale);
    int offset_x = (w - draw_w) / 2;
    int offset_y = (h - draw_h) / 2;

    double img_x = (click_x - offset_x) / scale;
    double img_y = (click_y - offset_y) / scale;

    return {
        std::clamp(static_cast<int>(img_x), 0, image.cols - 1),
        std::clamp(static_cast<int>(img_y), 0, image.rows - 1)
    };
}

void redraw_lines_and_refresh(GtkWidget* widget) {
    image.setTo(cv::Scalar(255, 255, 255));
    for (const auto& line : lines) {
        drawLineDDA(image, line.start, line.end, cv::Vec3b(0, 0, 255), line.thickness);
    }
    gtk_widget_queue_draw(widget);
}

gboolean draw_callback(GtkWidget* widget, cairo_t* cr, gpointer) {
    cv::Mat rgb_image;
    cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);

    GdkPixbuf* pixbuf = gdk_pixbuf_new_from_data(
        rgb_image.data, GDK_COLORSPACE_RGB, FALSE, 8,
        rgb_image.cols, rgb_image.rows, rgb_image.step, NULL, NULL);

    int w = gtk_widget_get_allocated_width(widget);
    int h = gtk_widget_get_allocated_height(widget);
    double scale_x = static_cast<double>(w) / image.cols;
    double scale_y = static_cast<double>(h) / image.rows;
    current_scale = std::min(scale_x, scale_y);

    int draw_w = static_cast<int>(image.cols * current_scale);
    int draw_h = static_cast<int>(image.rows * current_scale);
    int offset_x = (w - draw_w) / 2;
    int offset_y = (h - draw_h) / 2;

    cairo_translate(cr, offset_x, offset_y);
    cairo_scale(cr, current_scale, current_scale);

    gdk_cairo_set_source_pixbuf(cr, pixbuf, 0, 0);
    cairo_paint(cr);

    g_object_unref(pixbuf);
    return FALSE;
}

gboolean on_mouse_click(GtkWidget* widget, GdkEventButton* event, gpointer) {
    cv::Point pt = map_click_to_image(widget, event->x, event->y);
    const int threshold = 15;

    auto dist_to_segment = [](cv::Point p, cv::Point a, cv::Point b) -> double {
        cv::Point ab = b - a;
        double len_sq = ab.dot(ab);
        if (len_sq == 0) return cv::norm(p - a);
        double t = std::max(0.0, std::min(1.0, ((p - a).dot(ab)) / len_sq));
        cv::Point proj = a + t * ab;
        return cv::norm(p - proj);
    };

    if (event->button == 1) { // Left Click
        if (editMode != EditMode::None && selectedLineIndex >= 0) {
            if (selectedLineIndex < lines.size()) {
                auto& line = lines[selectedLineIndex];
                if (editMode == EditMode::MoveStart) line.start = pt;
                else if (editMode == EditMode::MoveEnd) line.end = pt;
                redraw_lines_and_refresh(widget);
            }
            editMode = EditMode::None;
            selectedLineIndex = -1;
        } else {
            bool found = false;
            for (int i = 0; i < lines.size(); ++i) {
                if (cv::norm(pt - lines[i].start) < threshold) {
                    editMode = EditMode::MoveStart;
                    selectedLineIndex = i;
                    found = true;
                    break;
                } else if (cv::norm(pt - lines[i].end) < threshold) {
                    editMode = EditMode::MoveEnd;
                    selectedLineIndex = i;
                    found = true;
                    break;
                }
            }
            if (!found) {
                if (!awaitingSecondClick) {
                    lineStart = pt;
                    awaitingSecondClick = true;
                } else {
                    lines.push_back({lineStart, pt, 1});
                    awaitingSecondClick = false;
                    redraw_lines_and_refresh(widget);
                }
            }
        }
    }
    else if (event->button == 3) { // Right Click (delete)
        auto it = std::find_if(lines.begin(), lines.end(), [&](const Line& l) {
            return dist_to_segment(pt, l.start, l.end) < threshold;
        });
        if (it != lines.end()) {
            lines.erase(it);
            redraw_lines_and_refresh(widget);
        }
    }
    else if (event->button == 2) { // Middle Click (wheel button: select line)
        auto it = std::find_if(lines.begin(), lines.end(), [&](const Line& l) {
            return dist_to_segment(pt, l.start, l.end) < threshold;
        });
    
        if (it != lines.end()) {
            selectedLineIndex = std::distance(lines.begin(), it);
            std::cout << "Line selected with middle click for potential thickness change." << std::endl;
        }
    }    

    return TRUE;
}


gboolean on_scroll(GtkWidget* widget, GdkEventScroll* event, gpointer) {
    double delta = 0;

    if (event->direction == GDK_SCROLL_SMOOTH) {
        gdouble dx = 0, dy = 0;
        gdk_event_get_scroll_deltas(reinterpret_cast<GdkEvent*>(event), &dx, &dy);
        delta = dy;
        std::cout << "Smooth scroll detected (dx=" << dx << ", dy=" << dy << ")\n";
    } else if (event->direction == GDK_SCROLL_UP) {
        delta = -1;
        std::cout << "Discrete scroll up detected\n";
    } else if (event->direction == GDK_SCROLL_DOWN) {
        delta = 1;
        std::cout << "Discrete scroll down detected\n";
    } else {
        std::cout << "Other scroll event detected\n";
    }

    if (delta != 0 && selectedLineIndex >= 0 && selectedLineIndex < lines.size()) {
        auto& line = lines[selectedLineIndex];

        if (delta < 0)
            line.thickness = std::min(50, line.thickness + 1);
        else
            line.thickness = std::max(1, line.thickness - 1);

        std::cout << "Changed selected line thickness to: " << line.thickness << "\n";
        redraw_lines_and_refresh(widget);
    }

    return TRUE;
}




int main(int argc, char* argv[]) {
    gtk_init(&argc, &argv);

    image = cv::Mat::zeros(600, 800, CV_8UC3);
    image.setTo(cv::Scalar(255, 255, 255));

    GtkWidget* window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_title(GTK_WINDOW(window), "Line Editor with Scaled Thickness");
    gtk_window_set_default_size(GTK_WINDOW(window), 1000, 800);

    drawing_area = gtk_drawing_area_new();
    gtk_widget_set_hexpand(drawing_area, TRUE);
    gtk_widget_set_vexpand(drawing_area, TRUE);

    gtk_container_add(GTK_CONTAINER(window), drawing_area);
    gtk_widget_add_events(
        drawing_area,
        GDK_BUTTON_PRESS_MASK |
        GDK_SCROLL_MASK |
        GDK_SMOOTH_SCROLL_MASK
    );

    g_signal_connect(drawing_area, "draw", G_CALLBACK(draw_callback), NULL);
    g_signal_connect(drawing_area, "button-press-event", G_CALLBACK(on_mouse_click), NULL);
    g_signal_connect(drawing_area, "scroll-event", G_CALLBACK(on_scroll), NULL);
    g_signal_connect(window, "destroy", G_CALLBACK(gtk_main_quit), NULL);

    gtk_widget_show_all(window);
    gtk_main();
    return 0;
}
