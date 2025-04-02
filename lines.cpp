#include <gtk/gtk.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

enum class ShapeType { Line, Circle };
ShapeType currentShape = ShapeType::Line;
GtkWidget* shape_menu;

struct Line {
    cv::Point start, end;
    int thickness;
};

struct Circle {
    cv::Point center;
    int radius;
};

enum class EditMode { None, MoveStart, MoveEnd };

std::vector<Line> lines;
std::vector<Circle> circles;
cv::Mat image;
GtkWidget* drawing_area;
cv::Point tempPoint;
bool awaitingSecondClick = false;
double current_scale = 1.0;

EditMode editMode = EditMode::None;
int selectedLineIndex = -1;
int selectedCircleIndex = -1;

void drawCircleMidpoint(cv::Mat& img, cv::Point center, int radius, const cv::Vec3b& color) {
    int x = 0, y = radius;
    int d = 1 - radius;

    auto drawSymmetricPoints = [&](int cx, int cy, int x, int y) {
        std::vector<cv::Point> points = {
            {cx + x, cy + y}, {cx - x, cy + y}, {cx + x, cy - y}, {cx - x, cy - y},
            {cx + y, cy + x}, {cx - y, cy + x}, {cx + y, cy - x}, {cx - y, cy - x}
        };
        for (auto& pt : points) {
            if (pt.x >= 0 && pt.x < img.cols && pt.y >= 0 && pt.y < img.rows)
                img.at<cv::Vec3b>(pt) = color;
        }
    };

    drawSymmetricPoints(center.x, center.y, x, y);
    while (x < y) {
        x++;
        if (d < 0) d += 2 * x + 1;
        else {
            y--;
            d += 2 * (x - y) + 1;
        }
        drawSymmetricPoints(center.x, center.y, x, y);
    }
}

void drawLineDDA(cv::Mat& img, cv::Point p1, cv::Point p2, const cv::Vec3b& color, int thickness = 1) {
    int dx = p2.x - p1.x;
    int dy = p2.y - p1.y;
    int steps = std::max(std::abs(dx), std::abs(dy));

    float x_inc = dx / static_cast<float>(steps);
    float y_inc = dy / static_cast<float>(steps);
    float x = p1.x, y = p1.y;

    int half = (thickness - 1) / 2;

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

void redraw_shapes(GtkWidget* widget) {
    image.setTo(cv::Scalar(255, 255, 255));
    for (const auto& line : lines)
        drawLineDDA(image, line.start, line.end, cv::Vec3b(0, 0, 255), line.thickness);
    for (const auto& circle : circles)
        drawCircleMidpoint(image, circle.center, circle.radius, cv::Vec3b(0, 255, 0));
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

bool try_select_circle(cv::Point pt, int threshold) {
    for (int i = 0; i < circles.size(); ++i) {
        if (std::abs(cv::norm(pt - circles[i].center) - circles[i].radius) < threshold) {
            selectedCircleIndex = i;
            selectedLineIndex = -1;
            return true;
        }
    }
    return false;
}

void try_delete_circle(cv::Point pt, GtkWidget* widget, int threshold) {
    auto it = std::find_if(circles.begin(), circles.end(), [&](const Circle& c) {
        return std::abs(cv::norm(pt - c.center) - c.radius) < threshold;
    });
    if (it != circles.end()) {
        circles.erase(it);
        redraw_shapes(widget);
    }
}

bool try_select_line_endpoint(cv::Point pt, int threshold) {
    for (int i = 0; i < lines.size(); ++i) {
        if (cv::norm(pt - lines[i].start) < threshold) {
            editMode = EditMode::MoveStart;
            selectedLineIndex = i;
            return true;
        } else if (cv::norm(pt - lines[i].end) < threshold) {
            editMode = EditMode::MoveEnd;
            selectedLineIndex = i;
            selectedCircleIndex = -1;
            return true;
        }
    }
    return false;
}

void try_delete_line(cv::Point pt, GtkWidget* widget, int threshold) {
    auto dist_to_segment = [](cv::Point p, cv::Point a, cv::Point b) -> double {
        cv::Point ab = b - a;
        double len_sq = ab.dot(ab);
        if (len_sq == 0) return cv::norm(p - a);
        double t = std::clamp((p - a).dot(ab) / len_sq, 0.0, 1.0);
        cv::Point proj = a + t * ab;
        return cv::norm(p - proj);
    };

    auto it = std::find_if(lines.begin(), lines.end(), [&](const Line& l) {
        return dist_to_segment(pt, l.start, l.end) < threshold;
    });
    if (it != lines.end()) {
        lines.erase(it);
        redraw_shapes(widget);
    }
}

void handle_circle_click(cv::Point pt, GdkEventButton* event, GtkWidget* widget, int threshold) {
    static bool movingCircle = false;

    if (event->button == 1) {
        if (movingCircle && selectedCircleIndex >= 0 && selectedCircleIndex < circles.size()) {
            circles[selectedCircleIndex].center = pt;
            movingCircle = false;
            selectedCircleIndex = -1;
            redraw_shapes(widget);
        } else if (try_select_circle(pt, threshold)) {
            movingCircle = true;
        } else {
            if (!awaitingSecondClick) {
                tempPoint = pt;
                awaitingSecondClick = true;
            } else {
                int radius = static_cast<int>(cv::norm(pt - tempPoint));
                if (radius > 0) {
                    circles.push_back({tempPoint, radius});
                }
                awaitingSecondClick = false;
                redraw_shapes(widget);
            }
        }
    } else if (event->button == 2) {
        if (try_select_circle(pt, threshold)) {
            std::cout << "Circle selected with middle click.\n";
        }
    } else if (event->button == 3) {
        try_delete_circle(pt, widget, threshold);
    }
}

void handle_line_click(cv::Point pt, GdkEventButton* event, GtkWidget* widget, int threshold) {
    static cv::Point lineStart;

    auto dist_to_segment = [](cv::Point p, cv::Point a, cv::Point b) -> double {
        cv::Point ab = b - a;
        double len_sq = ab.dot(ab);
        if (len_sq == 0) return cv::norm(p - a);
        double t = std::clamp((p - a).dot(ab) / len_sq, 0.0, 1.0);
        cv::Point proj = a + t * ab;
        return cv::norm(p - proj);
    };

    if (event->button == 1) {
        if (editMode != EditMode::None && selectedLineIndex >= 0 && selectedLineIndex < lines.size()) {
            auto& line = lines[selectedLineIndex];
            if (editMode == EditMode::MoveStart) line.start = pt;
            else if (editMode == EditMode::MoveEnd) line.end = pt;
            editMode = EditMode::None;
            selectedLineIndex = -1;
            redraw_shapes(widget);
        } else if (try_select_line_endpoint(pt, threshold)) {
            // selectedLineIndex & editMode are set
        } else {
            if (!awaitingSecondClick) {
                lineStart = pt;
                awaitingSecondClick = true;
            } else {
                if (pt != lineStart) {
                    lines.push_back({lineStart, pt, 1});
                }
                awaitingSecondClick = false;
                redraw_shapes(widget);
            }
        }
    } else if (event->button == 2) {
        auto it = std::find_if(lines.begin(), lines.end(), [&](const Line& l) {
            return dist_to_segment(pt, l.start, l.end) < threshold;
        });

        if (it != lines.end()) {
            selectedLineIndex = std::distance(lines.begin(), it);
            selectedCircleIndex = -1;
            std::cout << "Line selected with middle click for potential thickness change.\n";
        }
    } else if (event->button == 3) {
        try_delete_line(pt, widget, threshold);
    }
}

gboolean on_mouse_click(GtkWidget* widget, GdkEventButton* event, gpointer) {
    cv::Point pt = map_click_to_image(widget, event->x, event->y);
    const int threshold = 15;

    if (currentShape == ShapeType::Circle) {
        handle_circle_click(pt, event, widget, threshold);
    } else {
        handle_line_click(pt, event, widget, threshold);
    }

    return TRUE;
}

bool adjust_line_thickness(double delta) {
    if (selectedLineIndex >= 0 && selectedLineIndex < lines.size()) {
        auto& line = lines[selectedLineIndex];
        if (delta < 0)
            line.thickness = std::min(50, line.thickness + 1);
        else
            line.thickness = std::max(1, line.thickness - 1);
        std::cout << "Line thickness changed to: " << line.thickness << "\n";
        return true;
    }
    return false;
}

bool adjust_circle_radius(double delta) {
    if (selectedCircleIndex >= 0 && selectedCircleIndex < circles.size()) {
        auto& circle = circles[selectedCircleIndex];
        if (delta < 0)
            circle.radius = std::min(300, circle.radius + 5);
        else
            circle.radius = std::max(5, circle.radius - 5);
        std::cout << "Circle radius changed to: " << circle.radius << "\n";
        return true;
    }
    return false;
}

double extract_scroll_delta(GdkEventScroll* event) {
    double delta = 0;
    if (event->direction == GDK_SCROLL_SMOOTH) {
        gdouble dx = 0, dy = 0;
        gdk_event_get_scroll_deltas(reinterpret_cast<GdkEvent*>(event), &dx, &dy);
        delta = dy;
    } else if (event->direction == GDK_SCROLL_UP) {
        delta = -1;
    } else if (event->direction == GDK_SCROLL_DOWN) {
        delta = 1;
    }
    return delta;
}

gboolean on_scroll(GtkWidget* widget, GdkEventScroll* event, gpointer) {
    double delta = extract_scroll_delta(event);
    if (delta == 0) return TRUE;

    bool changed = false;

    // Prioritize circle editing
    if (adjust_circle_radius(delta)) {
        changed = true;
    } else if (adjust_line_thickness(delta)) {
        changed = true;
    }

    if (changed)
        redraw_shapes(widget);

    return TRUE;
}



void on_shape_selected(GtkComboBoxText* combo, gpointer) {
    const gchar* selected = gtk_combo_box_text_get_active_text(combo);
    if (g_strcmp0(selected, "Line") == 0) currentShape = ShapeType::Line;
    else if (g_strcmp0(selected, "Circle") == 0) currentShape = ShapeType::Circle;
    std::cout << "Shape changed to: " << selected << std::endl;
}

GtkWidget* create_shape_menu(GtkWidget* window) {
    GtkWidget* shape_menu_bar = gtk_menu_bar_new();
    GtkWidget* shape_menu_root = gtk_menu_item_new_with_label("Shapes");
    GtkWidget* shape_submenu = gtk_menu_new();

    GtkWidget* line_item = gtk_menu_item_new_with_label("Line");
    GtkWidget* circle_item = gtk_menu_item_new_with_label("Circle");

    gtk_menu_shell_append(GTK_MENU_SHELL(shape_submenu), line_item);
    gtk_menu_shell_append(GTK_MENU_SHELL(shape_submenu), circle_item);
    gtk_menu_item_set_submenu(GTK_MENU_ITEM(shape_menu_root), shape_submenu);
    gtk_menu_shell_append(GTK_MENU_SHELL(shape_menu_bar), shape_menu_root);

    // Callbacks for selection
    g_signal_connect(line_item, "activate", G_CALLBACK(+[](GtkWidget*, gpointer) {
        currentShape = ShapeType::Line;
        std::cout << "Shape changed to: Line" << std::endl;
    }), NULL);

    g_signal_connect(circle_item, "activate", G_CALLBACK(+[](GtkWidget*, gpointer) {
        currentShape = ShapeType::Circle;
        std::cout << "Shape changed to: Circle" << std::endl;
    }), NULL);

    return shape_menu_bar;
}



int main(int argc, char* argv[]) {
    gtk_init(&argc, &argv);
    image = cv::Mat::zeros(600, 800, CV_8UC3);
    image.setTo(cv::Scalar(255, 255, 255));
    GtkWidget* window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_title(GTK_WINDOW(window), "Line Editor with Scaled Thickness");
    gtk_window_set_default_size(GTK_WINDOW(window), 1000, 800);

    GtkWidget* vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    gtk_container_add(GTK_CONTAINER(window), vbox);

    shape_menu = create_shape_menu(window);
    gtk_box_pack_start(GTK_BOX(vbox), shape_menu, FALSE, FALSE, 0);


    drawing_area = gtk_drawing_area_new();
    gtk_widget_set_hexpand(drawing_area, TRUE);
    gtk_widget_set_vexpand(drawing_area, TRUE);
    gtk_box_pack_start(GTK_BOX(vbox), drawing_area, TRUE, TRUE, 0);

    gtk_widget_add_events(
        drawing_area,
        GDK_BUTTON_PRESS_MASK |
        GDK_SCROLL_MASK |
        GDK_SMOOTH_SCROLL_MASK);

    g_signal_connect(drawing_area, "draw", G_CALLBACK(draw_callback), NULL);
    g_signal_connect(drawing_area, "button-press-event", G_CALLBACK(on_mouse_click), NULL);
    g_signal_connect(drawing_area, "scroll-event", G_CALLBACK(on_scroll), NULL);
    g_signal_connect(window, "destroy", G_CALLBACK(gtk_main_quit), NULL);

    gtk_widget_show_all(window);
    gtk_main();
    return 0;
}
