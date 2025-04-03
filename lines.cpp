#include <gtk/gtk.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

enum class ShapeType { Line, Circle, Polygon }; // Added Polygon
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

struct Polygon {
    std::vector<cv::Point> vertices;
    int thickness;
};

enum class EditMode { None, MoveStart, MoveEnd };

std::vector<Line> lines;
std::vector<Circle> circles;
std::vector<Polygon> polygons; // New vector

cv::Mat image;
GtkWidget* drawing_area;
cv::Point tempPoint;
bool awaitingSecondClick = false;
double current_scale = 1.0;

EditMode editMode = EditMode::None;
int selectedLineIndex = -1;
int selectedCircleIndex = -1;
int selectedPolygonIndex = -1;

enum class PolygonMoveMode { None, MoveVertex, MoveEdge, MoveWhole };
PolygonMoveMode polygonMoveMode = PolygonMoveMode::None;
static cv::Point storedClick;
static bool waitingForSecondClick = false;

int selectedVertexIndex = -1;  // used for vertex or edge

std::vector<cv::Point> currentPolygonVertices; // in-progress polygon

bool use_antialiasing = false; 

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

double distance_to_line(cv::Point p, cv::Point a, cv::Point b) {
    cv::Point ab = b - a;
    double ab_len_sq = ab.dot(ab);

    if (ab_len_sq == 0.0)
        return cv::norm(p - a);  // a == b → treat as point

    double t = std::clamp((p - a).dot(ab) / ab_len_sq, 0.0, 1.0);
    cv::Point proj = a + t * ab;

    return cv::norm(p - proj);
}

cv::Vec3b lerp(const cv::Vec3b& a, const cv::Vec3b& b, float t) {
    cv::Vec3b result;
    for (int i = 0; i < 3; ++i)
        result[i] = static_cast<uchar>((1 - t) * a[i] + t * b[i]);
    return result;
}

float coverage(float thickness, float distance) {
    float half = thickness / 2.0f;
    float fade = std::max(0.0f, 1.0f - distance / half);
    return fade * fade;  // makes outer pixels dimmer
}

void plot(cv::Mat& img, int x, int y, float brightness, const cv::Vec3b& color) {
    if (x >= 0 && x < img.cols && y >= 0 && y < img.rows) {
        cv::Vec3b& pixel = img.at<cv::Vec3b>(y, x);
        pixel = lerp(pixel, color, brightness);
    }
}

void drawLineGuptaSproull(cv::Mat& img, cv::Point p1, cv::Point p2, const cv::Vec3b& color, float thickness = 1.0f) {
    bool steep = std::abs(p2.y - p1.y) > std::abs(p2.x - p1.x);
    
    if (steep) {
        std::swap(p1.x, p1.y);
        std::swap(p2.x, p2.y);
    }
    if (p1.x > p2.x) std::swap(p1, p2);

    int dx = p2.x - p1.x;
    int dy = p2.y - p1.y;

    float gradient = (dx == 0) ? 1.0f : dy / static_cast<float>(dx);
    float length = std::sqrt(dx * dx + dy * dy);
    if (length == 0) return;

    float invLength = 1.0f / length;
    float normal_x = steep ? -1 : gradient;
    float normal_y = steep ? gradient : 1;
    float normal_len = std::sqrt(normal_x * normal_x + normal_y * normal_y);
    normal_x /= normal_len;
    normal_y /= normal_len;

    auto intensify = [&](float fx, float fy, float dist) {
        float cov = coverage(thickness, std::abs(dist));
        if (cov > 0) {
            int ix = static_cast<int>(std::floor(fx + 0.5f));
            int iy = static_cast<int>(std::floor(fy + 0.5f));
            if (steep)
                plot(img, iy, ix, cov, color); // reverse
            else
                plot(img, ix, iy, cov, color);
        }
    };

    float x = static_cast<float>(p1.x);
    float y = static_cast<float>(p1.y);

    for (int i = 0; i <= dx; ++i) {
        // Central pixel
        intensify(x, y, 0);

        // Thickness simulation: perpendicular to line
        for (int j = 1; j <= thickness; ++j) {
            float offset_x = normal_x * j;
            float offset_y = normal_y * j;
            intensify(x + offset_x, y + offset_y, j);
            intensify(x - offset_x, y - offset_y, j);
        }

        x += 1.0f;
        y += gradient;
    }
}

void drawLineDDA(cv::Mat& img, cv::Point p1, cv::Point p2, const cv::Vec3b& color, int thickness = 1) {
    
    if(use_antialiasing)
    {
        drawLineGuptaSproull(image, p1, p2, color, thickness);
        return;
    }

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

void draw_in_progress_polygon(cv::Mat& img) {
    if (currentPolygonVertices.size() < 2) return;
    for (size_t i = 0; i < currentPolygonVertices.size() - 1; ++i) {
        drawLineDDA(img, currentPolygonVertices[i], currentPolygonVertices[i + 1], cv::Vec3b(200, 200, 0), 1);
    }
}

void drawPolygon(cv::Mat& img, const Polygon& poly, const cv::Vec3b& color) {
    if (poly.vertices.size() < 2) return;
    for (size_t i = 0; i < poly.vertices.size(); ++i) {
        cv::Point p1 = poly.vertices[i];
        cv::Point p2 = poly.vertices[(i + 1) % poly.vertices.size()];
        drawLineDDA(img, p1, p2, color, poly.thickness);
    }
}

void redraw_shapes(GtkWidget* widget) {
    image.setTo(cv::Scalar(255, 255, 255));
    for (const auto& line : lines)
        drawLineDDA(image, line.start, line.end, cv::Vec3b(0, 0, 255), line.thickness);
    for (const auto& circle : circles)
        drawCircleMidpoint(image, circle.center, circle.radius, cv::Vec3b(0, 255, 0));
    for (const auto& poly : polygons)
        drawPolygon(image, poly, cv::Vec3b(255, 0, 0));
    draw_in_progress_polygon(image);
    gtk_widget_queue_draw(widget);
}

double distance_to_segment(cv::Point p, cv::Point a, cv::Point b) {
    cv::Point ab = b - a;
    double len_sq = ab.dot(ab);
    if (len_sq == 0) return cv::norm(p - a);
    double t = std::clamp((p - a).dot(ab) / len_sq, 0.0, 1.0);
    cv::Point proj = a + t * ab;
    return cv::norm(p - proj);
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

void try_complete_polygon(cv::Point pt, GtkWidget* widget) {
    const int closeThreshold = 10;
    if (!currentPolygonVertices.empty() && cv::norm(pt - currentPolygonVertices[0]) < closeThreshold && currentPolygonVertices.size() >= 3) {
        polygons.push_back({currentPolygonVertices, 1});
        currentPolygonVertices.clear();
        redraw_shapes(widget);
    } else {
        currentPolygonVertices.push_back(pt);
        redraw_shapes(widget);
    }
}

void try_delete_polygon(cv::Point pt, GtkWidget* widget) {
    const int threshold = 10;
    auto dist_to_segment = [](cv::Point p, cv::Point a, cv::Point b) -> double {
        cv::Point ab = b - a;
        double len_sq = ab.dot(ab);
        if (len_sq == 0) return cv::norm(p - a);
        double t = std::clamp((p - a).dot(ab) / len_sq, 0.0, 1.0);
        cv::Point proj = a + t * ab;
        return cv::norm(p - proj);
    };
    for (auto it = polygons.begin(); it != polygons.end(); ++it) {
        const auto& vertices = it->vertices;
        for (size_t i = 0; i < vertices.size(); ++i) {
            cv::Point a = vertices[i];
            cv::Point b = vertices[(i + 1) % vertices.size()];
            if (dist_to_segment(pt, a, b) < threshold) {
                polygons.erase(it);
                redraw_shapes(widget);
                return;
            }
        }
    }
}

void try_select_polygon(cv::Point pt) {
    const int threshold = 10;
    auto dist_to_segment = [](cv::Point p, cv::Point a, cv::Point b) -> double {
        cv::Point ab = b - a;
        double len_sq = ab.dot(ab);
        if (len_sq == 0) return cv::norm(p - a);
        double t = std::clamp((p - a).dot(ab) / len_sq, 0.0, 1.0);
        cv::Point proj = a + t * ab;
        return cv::norm(p - proj);
    };
    for (int i = 0; i < polygons.size(); ++i) {
        const auto& vertices = polygons[i].vertices;
        for (size_t j = 0; j < vertices.size(); ++j) {
            cv::Point a = vertices[j];
            cv::Point b = vertices[(j + 1) % vertices.size()];
            if (dist_to_segment(pt, a, b) < threshold) {
                selectedPolygonIndex = i;
                selectedCircleIndex = -1;
                selectedLineIndex = -1;
                std::cout << "Polygon selected with middle click.\n";
                return;
            }
        }
    }
    selectedPolygonIndex = -1;
}

bool is_point_near_vertex(const std::vector<cv::Point>& vertices, cv::Point pt, int& outIndex, int threshold = 10) {
    for (int i = 0; i < vertices.size(); ++i) {
        if (cv::norm(pt - vertices[i]) < threshold) {
            outIndex = i;
            return true;
        }
    }
    return false;
}

bool is_point_near_edge_center(const std::vector<cv::Point>& vertices, cv::Point pt, int& outEdgeIndex, int threshold = 10) {
    for (int i = 0; i < vertices.size(); ++i) {
        cv::Point a = vertices[i];
        cv::Point b = vertices[(i + 1) % vertices.size()];
        cv::Point mid = (a + b) / 2;
        if (cv::norm(pt - mid) < threshold) {
            outEdgeIndex = i;
            return true;
        }
    }
    return false;
}

bool is_point_inside_polygon(const std::vector<cv::Point>& vertices, cv::Point pt) {
    return cv::pointPolygonTest(vertices, pt, false) >= 0;
}

void move_polygon_vertex(Polygon& poly, int vertexIndex, cv::Point newPos) {
    if (vertexIndex >= 0 && vertexIndex < poly.vertices.size()) {
        poly.vertices[vertexIndex] = newPos;
    }
}

void move_polygon_edge(Polygon& poly, int edgeStartIndex, cv::Point newMidpoint) {
    if (poly.vertices.size() < 2) return;
    cv::Point a = poly.vertices[edgeStartIndex];
    cv::Point b = poly.vertices[(edgeStartIndex + 1) % poly.vertices.size()];
    cv::Point currentMid = (a + b) / 2;
    cv::Point offset = newMidpoint - currentMid;
    poly.vertices[edgeStartIndex] += offset;
    poly.vertices[(edgeStartIndex + 1) % poly.vertices.size()] += offset;
}

void move_entire_polygon(Polygon& poly, cv::Point newPos) {
    cv::Point offset = newPos - storedClick;
    for (auto& v : poly.vertices)
        v += offset;
}

void handle_polygon_movement(cv::Point pt, GtkWidget* widget) {
    if (selectedPolygonIndex < 0 || selectedPolygonIndex >= polygons.size()) return;

    Polygon& poly = polygons[selectedPolygonIndex];

    switch (polygonMoveMode) {
        case PolygonMoveMode::MoveVertex:
            move_polygon_vertex(poly, selectedVertexIndex, pt);
            break;
        case PolygonMoveMode::MoveEdge:
            move_polygon_edge(poly, selectedVertexIndex, pt);
            break;
        case PolygonMoveMode::MoveWhole:
            move_entire_polygon(poly, pt);
            break;
        default:
            return;
    }

    // Reset state after move
    polygonMoveMode = PolygonMoveMode::None;
    selectedPolygonIndex = -1;
    selectedVertexIndex = -1;
    waitingForSecondClick = false;
    redraw_shapes(widget);
}

void handle_polygon_selection(cv::Point pt, GdkEventButton* event, GtkWidget* widget) {
    const int threshold = 10;

    for (int i = 0; i < polygons.size(); ++i) {
        const auto& poly = polygons[i];

        if ((event->state & GDK_CONTROL_MASK) && is_point_inside_polygon(poly.vertices, pt)) {
            polygonMoveMode = PolygonMoveMode::MoveWhole;
            selectedPolygonIndex = i;
            storedClick = pt;
            waitingForSecondClick = true;
            return;
        }

        int vertexIndex = -1;
        if (is_point_near_vertex(poly.vertices, pt, vertexIndex, threshold)) {
            polygonMoveMode = PolygonMoveMode::MoveVertex;
            selectedPolygonIndex = i;
            selectedVertexIndex = vertexIndex;
            waitingForSecondClick = true;
            return;
        }

        int edgeIndex = -1;
        if (is_point_near_edge_center(poly.vertices, pt, edgeIndex, threshold)) {
            polygonMoveMode = PolygonMoveMode::MoveEdge;
            selectedPolygonIndex = i;
            selectedVertexIndex = edgeIndex;
            waitingForSecondClick = true;
            return;
        }
    }

    // If nothing selected, assume vertex creation
    try_complete_polygon(pt, widget);

}

void handle_polygon_click(cv::Point pt, GdkEventButton* event, GtkWidget* widget) {
    if (waitingForSecondClick) {
        handle_polygon_movement(pt, widget);
    } else {
        handle_polygon_selection(pt, event, widget);
    }
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

    switch (currentShape) {
        case ShapeType::Polygon:
            if (event->button == 1) {
                handle_polygon_click(pt, event, widget);
            } else if (event->button == 2) {
                try_select_polygon(pt);
            } else if (event->button == 3) {
                try_delete_polygon(pt, widget);
            }
            break;
        case ShapeType::Circle:
            handle_circle_click(pt, event, widget, threshold);
            break;
        case ShapeType::Line:
            handle_line_click(pt, event, widget, threshold);
            break;
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

    // Polygon thickness
    if (selectedPolygonIndex >= 0 && selectedPolygonIndex < polygons.size()) {
        auto& poly = polygons[selectedPolygonIndex];
        if (delta < 0)
            poly.thickness = std::min(50, poly.thickness + 1);
        else
            poly.thickness = std::max(1, poly.thickness - 1);
        std::cout << "Polygon thickness changed to: " << poly.thickness << "\n";
        changed = true;
    }

    // Circle radius
    else if (selectedCircleIndex >= 0 && selectedCircleIndex < circles.size()) {
        auto& circle = circles[selectedCircleIndex];
        if (delta < 0)
            circle.radius = std::min(300, circle.radius + 5);
        else
            circle.radius = std::max(5, circle.radius - 5);
        std::cout << "Circle radius changed to: " << circle.radius << "\n";
        changed = true;
    }

    // Line thickness
    else if (selectedLineIndex >= 0 && selectedLineIndex < lines.size()) {
        auto& line = lines[selectedLineIndex];
        if (delta < 0)
            line.thickness = std::min(50, line.thickness + 1);
        else
            line.thickness = std::max(1, line.thickness - 1);
        std::cout << "Line thickness changed to: " << line.thickness << "\n";
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

void clear_all_shapes(GtkWidget* widget) {
    lines.clear();
    circles.clear();
    polygons.clear();
    currentPolygonVertices.clear();
    selectedLineIndex = -1;
    selectedCircleIndex = -1;
    selectedPolygonIndex = -1;
    awaitingSecondClick = false;
    polygonMoveMode = PolygonMoveMode::None;
    editMode = EditMode::None;

    std::cout << "Canvas cleared.\n";
    redraw_shapes(widget);
}



void save_shapes_to_file(GtkWidget* parent) {
    GtkWidget* dialog = gtk_file_chooser_dialog_new("Save Shapes",
        GTK_WINDOW(parent),
        GTK_FILE_CHOOSER_ACTION_SAVE,
        "_Cancel", GTK_RESPONSE_CANCEL,
        "_Save", GTK_RESPONSE_ACCEPT,
        NULL);

    gtk_file_chooser_set_do_overwrite_confirmation(GTK_FILE_CHOOSER(dialog), TRUE);
    gtk_file_chooser_set_current_name(GTK_FILE_CHOOSER(dialog), "shapes.vec");

    if (gtk_dialog_run(GTK_DIALOG(dialog)) == GTK_RESPONSE_ACCEPT) {
        char* filename = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dialog));
        std::ofstream out(filename);
        if (!out) {
            std::cerr << "Failed to open file for saving.\n";
            g_free(filename);
            gtk_widget_destroy(dialog);
            return;
        }

        for (const auto& line : lines)
            out << "LINE " << line.start.x << " " << line.start.y << " "
                        << line.end.x << " " << line.end.y << " "
                        << line.thickness << "\n";

        for (const auto& circle : circles)
            out << "CIRCLE " << circle.center.x << " " << circle.center.y << " "
                           << circle.radius << "\n";

        for (const auto& poly : polygons) {
            out << "POLYGON " << poly.thickness << " " << poly.vertices.size();
            for (const auto& v : poly.vertices)
                out << " " << v.x << " " << v.y;
            out << "\n";
        }

        out.close();
        g_free(filename);
    }

    gtk_widget_destroy(dialog);
}

void load_shapes_from_file(GtkWidget* parent) {
    GtkWidget* dialog = gtk_file_chooser_dialog_new("Load Shapes",
        GTK_WINDOW(parent),
        GTK_FILE_CHOOSER_ACTION_OPEN,
        "_Cancel", GTK_RESPONSE_CANCEL,
        "_Open", GTK_RESPONSE_ACCEPT,
        NULL);

    if (gtk_dialog_run(GTK_DIALOG(dialog)) == GTK_RESPONSE_ACCEPT) {
        char* filename = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dialog));
        std::ifstream in(filename);
        if (!in) {
            std::cerr << "Failed to open file for loading.\n";
            g_free(filename);
            gtk_widget_destroy(dialog);
            return;
        }

        // Clear current shapes
        lines.clear();
        circles.clear();
        polygons.clear();
        currentPolygonVertices.clear();

        std::string type;
        while (in >> type) {
            if (type == "LINE") {
                Line l;
                in >> l.start.x >> l.start.y >> l.end.x >> l.end.y >> l.thickness;
                lines.push_back(l);
            } else if (type == "CIRCLE") {
                Circle c;
                in >> c.center.x >> c.center.y >> c.radius;
                circles.push_back(c);
            } else if (type == "POLYGON") {
                Polygon p;
                int count;
                in >> p.thickness >> count;
                for (int i = 0; i < count; ++i) {
                    cv::Point pt;
                    in >> pt.x >> pt.y;
                    p.vertices.push_back(pt);
                }
                polygons.push_back(p);
            }
        }

        in.close();
        redraw_shapes(parent);
        g_free(filename);
    }

    gtk_widget_destroy(dialog);
}

GtkWidget* create_shape_menu(GtkWidget* window) {
    GtkWidget* menu_bar = gtk_menu_bar_new();

    // === FILE MENU ===
    GtkWidget* file_menu_root = gtk_menu_item_new_with_label("File");
    GtkWidget* file_submenu = gtk_menu_new();
    GtkWidget* clear_item = gtk_menu_item_new_with_label("Clear");

    gtk_menu_shell_append(GTK_MENU_SHELL(file_submenu), clear_item);
    gtk_menu_item_set_submenu(GTK_MENU_ITEM(file_menu_root), file_submenu);
    gtk_menu_shell_append(GTK_MENU_SHELL(menu_bar), file_menu_root);

    // === SHAPES MENU ===
    GtkWidget* shape_menu_root = gtk_menu_item_new_with_label("Shapes");
    GtkWidget* shape_submenu = gtk_menu_new();

    GtkWidget* line_item = gtk_menu_item_new_with_label("Line");
    GtkWidget* circle_item = gtk_menu_item_new_with_label("Circle");
    GtkWidget* polygon_item = gtk_menu_item_new_with_label("Polygon");
    GtkWidget* aa_toggle_item = gtk_check_menu_item_new_with_label("Enable Anti-Aliasing");
    GtkWidget* save_item = gtk_menu_item_new_with_label("Save");
    GtkWidget* load_item = gtk_menu_item_new_with_label("Load");

    gtk_menu_shell_append(GTK_MENU_SHELL(shape_submenu), line_item);
    gtk_menu_shell_append(GTK_MENU_SHELL(shape_submenu), circle_item);
    gtk_menu_shell_append(GTK_MENU_SHELL(shape_submenu), polygon_item);
    gtk_menu_shell_append(GTK_MENU_SHELL(shape_submenu), aa_toggle_item);
    gtk_menu_shell_append(GTK_MENU_SHELL(file_submenu), save_item);
    gtk_menu_shell_append(GTK_MENU_SHELL(file_submenu), load_item);

    gtk_menu_item_set_submenu(GTK_MENU_ITEM(shape_menu_root), shape_submenu);
    gtk_menu_shell_append(GTK_MENU_SHELL(menu_bar), shape_menu_root);

    // === SIGNALS ===

    g_signal_connect(line_item, "activate", G_CALLBACK(+[](GtkWidget*, gpointer) {
        currentShape = ShapeType::Line;
        selectedCircleIndex = -1;
        selectedPolygonIndex = -1;
        std::cout << "Shape changed to: Line" << std::endl;
    }), NULL);

    g_signal_connect(circle_item, "activate", G_CALLBACK(+[](GtkWidget*, gpointer) {
        currentShape = ShapeType::Circle;
        selectedLineIndex = -1;
        selectedPolygonIndex = -1;
        std::cout << "Shape changed to: Circle" << std::endl;
    }), NULL);

    g_signal_connect(polygon_item, "activate", G_CALLBACK(+[](GtkWidget*, gpointer) {
        currentShape = ShapeType::Polygon;
        selectedLineIndex = -1;
        selectedPolygonIndex = -1;
        std::cout << "Shape changed to: Polygon" << std::endl;
    }), NULL);

    g_signal_connect(aa_toggle_item, "toggled", G_CALLBACK(+[](GtkCheckMenuItem* item, gpointer) {
        use_antialiasing = gtk_check_menu_item_get_active(item);
        std::cout << "Anti-aliasing: " << (use_antialiasing ? "ON" : "OFF") << std::endl;
        redraw_shapes(drawing_area);
    }), NULL);

    g_signal_connect(clear_item, "activate", G_CALLBACK(+[](GtkWidget*, gpointer) {
        clear_all_shapes(drawing_area);
    }), NULL);

    g_signal_connect(save_item, "activate", G_CALLBACK(+[](GtkWidget*, gpointer) {
        save_shapes_to_file(gtk_widget_get_toplevel(drawing_area));
    }), NULL);
    
    g_signal_connect(load_item, "activate", G_CALLBACK(+[](GtkWidget*, gpointer) {
        load_shapes_from_file(gtk_widget_get_toplevel(drawing_area));
    }), NULL);
    

    return menu_bar;
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
