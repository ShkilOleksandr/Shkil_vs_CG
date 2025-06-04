// scene_with_cubes_cylinders_spheres_cones.cpp
//
// A GTK/OpenCV application that lets you place:
//   - Cubes     (triangle-mesh)
//   - Cylinders  (triangle-mesh)
//   - Spheres    (triangle-mesh, latitude/longitude)
//   - Cones      (triangle-mesh)
// Supports interactive placement, selection, rotation (no modifier), scaling (Shift+scroll), and scene load/save.
//
// Compile (Linux):
//   g++ scene_with_cubes_cylinders_spheres_cones.cpp -o scene_app `pkg-config --cflags --libs gtk+-3.0 opencv4`
//
// Run:
//   ./scene_app [optional_scene.scene]
//
// If you pass a ".scene" file, it will load cubes, cylinders, spheres, and cones.

#include <cmath>
#include <fstream>
#include <gtk/gtk.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <array>
#include <limits>
#include <opencv2/opencv.hpp>
using namespace std;

//————————————————————————————————————————————————————————————————————
// GLOBAL ENUMS & CONSTANTS

enum class ShapeType { Cube, Cylinder, Sphere, Cone };
ShapeType currentShape = ShapeType::Cube;

constexpr float DEFAULT_CUBE_SIZE   = 50.0f;
constexpr float DEFAULT_CYL_RADIUS  = 25.0f;
constexpr float DEFAULT_CYL_HEIGHT  = 50.0f;
constexpr int   DEFAULT_CYL_SUBDIV  = 16;

constexpr float DEFAULT_SPH_RADIUS    = 30.0f;
constexpr int   DEFAULT_SPH_LAT_BANDS = 16;
constexpr int   DEFAULT_SPH_LON_BANDS = 16;

constexpr float DEFAULT_CONE_RADIUS   = 25.0f;
constexpr float DEFAULT_CONE_HEIGHT   = 50.0f;
constexpr int   DEFAULT_CONE_SUBDIV   = 16;

// selectedShapeIndex:
//   -1                      → no shape selected.
//   [0 .. cubes.size()-1]                                 → index into cubes
//   [cubes.size() .. cubes.size()+cylinders.size()-1]     → index-cubes.size() into cylinders
//   [cubes.size()+cylinders.size() .. cubes+…+cylinders+spheres.size()-1]
//       → index-(cubes.size()+cylinders.size()) into spheres
//   [cubes.size()+cylinders.size()+spheres.size() .. … + cones.size()-1]
//       → index-(cubes.size()+cylinders.size()+spheres.size()) into cones
int selectedShapeIndex = -1;

enum class Axis { X, Y, Z };
Axis currentAxis = Axis::Z;
bool use_antialiasing = false;

static int clampInt(int x, int low, int high) {
    return (x < low ? low : (x > high ? high : x));
}

//————————————————————————————————————————————————————————————————————
// FORWARD DECLARATIONS

bool loadScene(const std::string& filename);
bool saveScene(const std::string& filename);

//————————————————————————————————————————————————————————————————————
// CUBE STRUCT (MESH)

struct Cube {
    cv::Point3f                 center;
    float                       sideLength;
    cv::Vec3b                   color       = {0,255,0};
    cv::Matx33f                 orientation = cv::Matx33f::eye();

    // Exactly 8 vertices + 12 triangular faces:
    std::vector<cv::Point3f>    vertices;  // 8 corners
    std::vector<std::array<int,3>> faces;   // 12 triangles

    Cube(const cv::Point3f &c, float s,
         const cv::Matx33f &ori = cv::Matx33f::eye(),
         const cv::Vec3b   &col = {0,255,0})
      : center(c), sideLength(s), orientation(ori), color(col)
    {
        buildMesh();
    }

    void rotateX(float angle) {
        float c = cosf(angle), s = sinf(angle);
        cv::Matx33f Rx(
            1,  0,  0,
            0,  c, -s,
            0,  s,  c
        );
        orientation = Rx * orientation;
        buildMesh();
    }
    void rotateY(float angle) {
        float c = cosf(angle), s = sinf(angle);
        cv::Matx33f Ry(
            c,  0,  s,
            0,  1,  0,
           -s,  0,  c
        );
        orientation = Ry * orientation;
        buildMesh();
    }
    void rotateZ(float angle) {
        float c = cosf(angle), s = sinf(angle);
        cv::Matx33f Rz(
           c, -s, 0,
           s,  c, 0,
           0,  0, 1
        );
        orientation = Rz * orientation;
        buildMesh();
    }

    void buildMesh() {
    float h = sideLength * 0.5f;

    // REMOVE STATIC KEYWORD HERE TO FIX VERTEX UPDATE ISSUE
    const cv::Point3f baseCorners[8] = {
        {-h,-h,-h}, {+h,-h,-h}, {+h,+h,-h}, {-h,+h,-h},
        {-h,-h,+h}, {+h,-h,+h}, {+h,+h,+h}, {-h,+h,+h}
    };

    vertices.clear();
    vertices.reserve(8);
    for (int i = 0; i < 8; ++i) {
        cv::Vec3f v(baseCorners[i].x, baseCorners[i].y, baseCorners[i].z);
        cv::Vec3f vr = orientation * v;
        vertices.emplace_back(
            center.x + vr[0],
            center.y + vr[1],
            center.z + vr[2]
        );
    }

    // ALWAYS REBUILD FACES EACH TIME
    faces.clear();
    faces.reserve(12);
    faces.push_back({0,1,2}); faces.push_back({0,2,3});
    faces.push_back({4,6,5}); faces.push_back({4,7,6});
    faces.push_back({1,5,6}); faces.push_back({1,6,2});
    faces.push_back({0,7,3}); faces.push_back({0,4,7});
    faces.push_back({0,5,4}); faces.push_back({0,1,5});
    faces.push_back({3,6,2}); faces.push_back({3,7,6});
    }


    void scale(float factor) {
        sideLength *= factor;
        buildMesh();
    }
};

//————————————————————————————————————————————————————————————————————
// CYLINDER STRUCT (TRIANGULAR MESH)

struct Cylinder {
    cv::Point3f                 center;
    float                       radius;
    float                       height;
    int                         subdivisions;
    cv::Vec3b                   color       = {0,255,0};
    cv::Matx33f                 orientation = cv::Matx33f::eye();
    std::vector<cv::Point3f>    vertices;    // 2*N circle points + 2 centers
    std::vector<std::array<int,3>> faces;    // list of triangular faces

    Cylinder(const cv::Point3f &c, float r, float h, int sub,
             const cv::Matx33f &ori = cv::Matx33f::eye(),
             const cv::Vec3b   &col = {0,255,0})
      : center(c), radius(r), height(h), subdivisions(sub),
        orientation(ori), color(col)
    {
        buildMesh();
    }

    void rotateX(float angle) {
        float c = cosf(angle), s = sinf(angle);
        cv::Matx33f Rx(
            1,  0,  0,
            0,  c, -s,
            0,  s,  c
        );
        orientation = Rx * orientation;
        buildMesh();
    }
    void rotateY(float angle) {
        float c = cosf(angle), s = sinf(angle);
        cv::Matx33f Ry(
            c,  0,  s,
            0,  1,  0,
           -s,  0,  c
        );
        orientation = Ry * orientation;
        buildMesh();
    }
    void rotateZ(float angle) {
        float c = cosf(angle), s = sinf(angle);
        cv::Matx33f Rz(
           c, -s, 0,
           s,  c, 0,
           0,  0, 1
        );
        orientation = Rz * orientation;
        buildMesh();
    }

    void buildMesh() {
        vertices.clear();
        faces.clear();

        float halfh = height * 0.5f;
        // 1) Generate N bottom-circle points (z = -halfh), N top-circle points (z = +halfh)
        for (int i = 0; i < subdivisions; ++i) {
            float theta = (2.0f * float(M_PI) * i) / subdivisions;
            float x = radius * cosf(theta);
            float y = radius * sinf(theta);
            vertices.push_back({ x, y, -halfh });
        }
        for (int i = 0; i < subdivisions; ++i) {
            float theta = (2.0f * float(M_PI) * i) / subdivisions;
            float x = radius * cosf(theta);
            float y = radius * sinf(theta);
            vertices.push_back({ x, y, +halfh });
        }
        int idxBottomCenter = 2*subdivisions;
        int idxTopCenter    = 2*subdivisions + 1;
        vertices.push_back({ 0.0f, 0.0f, -halfh });
        vertices.push_back({ 0.0f, 0.0f, +halfh });

        // 2) Bottom cap (fan)
        for (int i = 0; i < subdivisions; ++i) {
            int iNext = (i + 1) % subdivisions;
            faces.push_back({ idxBottomCenter, iNext, i });
        }
        // 3) Top cap (fan)
        for (int i = 0; i < subdivisions; ++i) {
            int iNext   = (i + 1) % subdivisions;
            int topI    = i + subdivisions;
            int topNext = iNext + subdivisions;
            faces.push_back({ idxTopCenter, topI, topNext });
        }
        // 4) Side faces: each quad between (i, i+1) on bottom and (i, i+1)+subdiv on top
        for (int i = 0; i < subdivisions; ++i) {
            int iNext   = (i + 1) % subdivisions;
            int botI    = i;
            int botNext = iNext;
            int topI    = i + subdivisions;
            int topNext = iNext + subdivisions;
            // Triangle 1: (botI, botNext, topNext)
            faces.push_back({ botI, botNext, topNext });
            // Triangle 2: (botI, topNext, topI)
            faces.push_back({ botI, topNext, topI });
        }

        // 5) Apply orientation + translation
        for (auto &v : vertices) {
            cv::Vec3f vv(v.x, v.y, v.z);
            cv::Vec3f vr = orientation * vv;
            v = { center.x + vr[0], center.y + vr[1], center.z + vr[2] };
        }
    }

    void scale(float factor) {
        radius *= factor;
        height *= factor;
        buildMesh();
    }
};

//————————————————————————————————————————————————————————————————————
// SPHERE STRUCT (LAT/LON TRIANGLE MESH)

struct Sphere {
    cv::Point3f              center;
    float                    radius;
    int                      latBands;
    int                      lonBands;
    cv::Vec3b                color       = {0,255,0};
    cv::Matx33f              orientation = cv::Matx33f::eye();
    std::vector<cv::Point3f> vertices;   // (latBands+1) × lonBands points
    std::vector<std::array<int,3>> faces;

    Sphere(const cv::Point3f &c, float r, int latB, int lonB,
           const cv::Matx33f &ori = cv::Matx33f::eye(),
           const cv::Vec3b   &col = {0,255,0})
      : center(c), radius(r), latBands(latB), lonBands(lonB),
        orientation(ori), color(col)
    {
        buildMesh();
    }

    void rotateX(float angle) {
        float c = cosf(angle), s = sinf(angle);
        cv::Matx33f Rx(
            1,  0,  0,
            0,  c, -s,
            0,  s,  c
        );
        orientation = Rx * orientation;
        buildMesh();
    }
    void rotateY(float angle) {
        float c = cosf(angle), s = sinf(angle);
        cv::Matx33f Ry(
            c,  0,  s,
            0,  1,  0,
           -s,  0,  c
        );
        orientation = Ry * orientation;
        buildMesh();
    }
    void rotateZ(float angle) {
        float c = cosf(angle), s = sinf(angle);
        cv::Matx33f Rz(
           c, -s, 0,
           s,  c, 0,
           0,  0, 1
        );
        orientation = Rz * orientation;
        buildMesh();
    }

    void buildMesh() {
        vertices.clear();
        faces.clear();

        // 1) Generate all (latBands+1) × lonBands vertices
        for (int i = 0; i <= latBands; ++i) {
            float theta = (float(i) * float(M_PI)) / float(latBands);
            float sinT  = sinf(theta);
            float cosT  = cosf(theta);
            for (int j = 0; j < lonBands; ++j) {
                float phi = (float(j) * 2.0f * float(M_PI)) / float(lonBands);
                float sinP = sinf(phi);
                float cosP = cosf(phi);
                cv::Point3f vLocal = {
                    radius * sinT * cosP,   // x
                    radius * sinT * sinP,   // y
                    radius * cosT           // z
                };
                // Apply orientation:
                cv::Vec3f vv(vLocal.x, vLocal.y, vLocal.z);
                cv::Vec3f vr = orientation * vv;
                vertices.push_back({ center.x + vr[0],
                                     center.y + vr[1],
                                     center.z + vr[2] });
            }
        }

        // 2) Build faces (two triangles per “quad” between lat i..i+1 and lon j..j+1)
        for (int i = 0; i < latBands; ++i) {
            for (int j = 0; j < lonBands; ++j) {
                int nextJ = (j + 1) % lonBands;
                int v00 = i * lonBands + j;            // (i, j)
                int v01 = i * lonBands + nextJ;        // (i, j+1)
                int v10 = (i + 1) * lonBands + j;      // (i+1, j)
                int v11 = (i + 1) * lonBands + nextJ;  // (i+1, j+1)

                // Two triangles per quad:
                faces.push_back({ v00, v10, v11 });
                faces.push_back({ v00, v11, v01 });
            }
        }
    }

    void scale(float factor) {
        radius *= factor;
        buildMesh();
    }
};

//————————————————————————————————————————————————————————————————————
// CONE STRUCT (TRIANGULAR MESH)

struct Cone {
    cv::Point3f                 center;
    float                       radius;
    float                       height;
    int                         subdivisions;
    cv::Vec3b                   color       = {0,255,0};
    cv::Matx33f                 orientation = cv::Matx33f::eye();
    std::vector<cv::Point3f>    vertices;    // N base-circle points + apex + base-center
    std::vector<std::array<int,3>> faces;    // triangular faces

    Cone(const cv::Point3f &c, float r, float h, int sub,
         const cv::Matx33f &ori = cv::Matx33f::eye(),
         const cv::Vec3b   &col = {0,255,0})
      : center(c), radius(r), height(h), subdivisions(sub),
        orientation(ori), color(col)
    {
        buildMesh();
    }

    void rotateX(float angle) {
        float c = cosf(angle), s = sinf(angle);
        cv::Matx33f Rx(
            1,  0,  0,
            0,  c, -s,
            0,  s,  c
        );
        orientation = Rx * orientation;
        buildMesh();
    }
    void rotateY(float angle) {
        float c = cosf(angle), s = sinf(angle);
        cv::Matx33f Ry(
            c,  0,  s,
            0,  1,  0,
           -s,  0,  c
        );
        orientation = Ry * orientation;
        buildMesh();
    }
    void rotateZ(float angle) {
        float c = cosf(angle), s = sinf(angle);
        cv::Matx33f Rz(
           c, -s, 0,
           s,  c, 0,
           0,  0, 1
        );
        orientation = Rz * orientation;
        buildMesh();
    }

    void buildMesh() {
        vertices.clear();
        faces.clear();

        float halfh = height * 0.5f;
        // Apex at +halfh, base circle at –halfh
        // 1) Generate N base-circle points (z = –halfh)
        for (int i = 0; i < subdivisions; ++i) {
            float theta = (2.0f * float(M_PI) * i) / subdivisions;
            float x = radius * cosf(theta);
            float y = radius * sinf(theta);
            vertices.push_back({ x, y, -halfh });
        }
        // 2) Apex point (index = subdivisions)
        int idxApex = subdivisions;
        vertices.push_back({ 0.0f, 0.0f, +halfh });
        // 3) Base center (for capping, index = subdivisions+1)
        int idxBaseCenter = subdivisions + 1;
        vertices.push_back({ 0.0f, 0.0f, -halfh });

        // 4) Side faces: each base edge → triangle to apex
        for (int i = 0; i < subdivisions; ++i) {
            int iNext = (i + 1) % subdivisions;
            faces.push_back({ i, iNext, idxApex });
        }
        // 5) Base cap (fan around base center)
        for (int i = 0; i < subdivisions; ++i) {
            int iNext = (i + 1) % subdivisions;
            faces.push_back({ idxBaseCenter, iNext, i });
        }

        // 6) Apply orientation + translation
        for (auto &v : vertices) {
            cv::Vec3f vv(v.x, v.y, v.z);
            cv::Vec3f vr = orientation * vv;
            v = { center.x + vr[0], center.y + vr[1], center.z + vr[2] };
        }
    }

    void scale(float factor) {
        radius *= factor;
        height *= factor;
        buildMesh();
    }
};

//————————————————————————————————————————————————————————————————————
// GLOBAL CONTAINERS

vector<Cube>      cubes;
vector<Cylinder>  cylinders;
vector<Sphere>    spheres;
vector<Cone>      cones;

//————————————————————————————————————————————————————————————————————
// OFFSCREEN “image” + GTK DRAWING AREA

cv::Mat    image;
GtkWidget *drawing_area;
double     current_scale = 1.0;

//————————————————————————————————————————————————————————————————————
// FORWARD DECLARATIONS (callbacks, menu builders)

GtkWidget* create_main_menu();
cv::Point  map_click_to_image(GtkWidget *widget, double click_x, double click_y);
void       drawLineDDA(cv::Mat &img, cv::Point p0, cv::Point p1,
                       const cv::Vec3b &color, int thickness = 1);

void drawCubeMesh(cv::Mat &img, const Cube &cube);
void drawCylinderMesh(cv::Mat &img, const Cylinder &cyl);
void drawSphereMesh(cv::Mat &img, const Sphere &sph);
void drawConeMesh(cv::Mat &img, const Cone &cone);

gboolean on_mouse_click(GtkWidget *widget, GdkEventButton *event, gpointer);
gboolean on_scroll(GtkWidget *widget, GdkEventScroll *event, gpointer);
gboolean on_key_press(GtkWidget*, GdkEventKey *ev, gpointer);
gboolean on_key_release(GtkWidget*, GdkEventKey *ev, gpointer);
gboolean draw_callback(GtkWidget *widget, cairo_t *cr, gpointer);

static void on_cube_activate(GtkMenuItem*, gpointer);
static void on_cylinder_activate(GtkMenuItem*, gpointer);
static void on_sphere_activate(GtkMenuItem*, gpointer);
static void on_cone_activate(GtkMenuItem*, gpointer);
static void on_aliasing_toggle(GtkMenuItem*, gpointer);
static void on_load_scene_activate(GtkMenuItem*, gpointer);
static void on_save_scene_activate(GtkMenuItem*, gpointer);

//————————————————————————————————————————————————————————————————————
// UTILITY: MAP GTK CLICK → IMAGE PIXEL COORDS

cv::Point map_click_to_image(GtkWidget *widget, double click_x, double click_y) {
    int w = gtk_widget_get_allocated_width(widget);
    int h = gtk_widget_get_allocated_height(widget);
    double sx = double(w) / image.cols;
    double sy = double(h) / image.rows;
    double scale = std::min(sx, sy);

    int draw_w = int(image.cols * scale);
    int draw_h = int(image.rows * scale);
    int ox = (w - draw_w) / 2;
    int oy = (h - draw_h) / 2;

    double img_x = (click_x - ox) / scale;
    double img_y = (click_y - oy) / scale;
    return {
        std::clamp(int(img_x), 0, image.cols - 1),
        std::clamp(int(img_y), 0, image.rows - 1)
    };
}

//————————————————————————————————————————————————————————————————————
// ANTIALIASED LINE DRAWING (Gupta-Sproull + DDA)

double distance_to_line(cv::Point p, cv::Point a, cv::Point b) {
    cv::Point ab = b - a;
    double ab_len_sq = ab.dot(ab);
    if (ab_len_sq == 0.0) return cv::norm(p - a);
    double t = std::clamp((p - a).dot(ab) / ab_len_sq, 0.0, 1.0);
    cv::Point proj = a + t*ab;
    return cv::norm(p - proj);
}

float coverage(float thickness, float distance) {
    float radius = thickness / 2.0f;
    float t = distance / radius;
    return std::exp(-t*t*0.75f);
}

void plot(cv::Mat &img, int x, int y, float alpha, const cv::Vec3b &color) {
    if (x < 0 || x >= img.cols || y < 0 || y >= img.rows) return;
    cv::Vec3b &p = img.at<cv::Vec3b>(y, x);
    for (int i = 0; i < 3; ++i) {
        int blended = int(p[i]*(1.0f - alpha) + color[i]*alpha + 0.5f);
        p[i] = uchar(std::clamp(blended, 0, 255));
    }
}

void drawLineGuptaSproull(cv::Mat &img, cv::Point p0, cv::Point p1,
                          const cv::Vec3b &color, float thickness = 1.0f)
{
    auto plotAA = [&](int x, int y, float d) {
        float alpha = coverage(thickness, d);
        if (alpha > 0.0f) plot(img, x, y, alpha, color);
    };
    bool steep = std::abs(p1.y - p0.y) > std::abs(p1.x - p0.x);
    if (steep) {
        std::swap(p0.x, p0.y);
        std::swap(p1.x, p1.y);
    }
    if (p0.x > p1.x) std::swap(p0, p1);

    int dx = p1.x - p0.x;
    int dy = p1.y - p0.y;
    float gradient = (dx == 0) ? 0.0f : float(dy) / dx;
    float intery = float(p0.y) + gradient;

    float radius = thickness / 2.0f;
    int coverageRange = int(std::ceil(radius)) + 1;

    for (int x = p0.x + 1; x <= p1.x - 1; ++x) {
        int ycenter = int(intery);
        float frac = intery - ycenter;
        for (int k = -coverageRange; k <= coverageRange; ++k) {
            float d = std::abs(k - frac);
            if (steep)      plotAA(ycenter + k, x, d);
            else            plotAA(x, ycenter + k, d);
        }
        intery += gradient;
    }
}

inline void setPixel(cv::Mat &img, int x, int y, const cv::Vec3b &color) {
    if (x < 0 || x >= img.cols || y < 0 || y >= img.rows) return;
    img.at<cv::Vec3b>(y, x) = color;
}

inline float smoothstep(float edge0, float edge1, float x) {
    float t = std::clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t*t*(3.0f - 2.0f*t);
}

void drawLineThickAA(cv::Mat &img, cv::Point p0, cv::Point p1,
                     const cv::Vec3b &color, int thickness = 1, bool antialias = false)
{
    cv::Point2f dir = cv::Point2f(p1) - cv::Point2f(p0);
    float len = std::hypot(dir.x, dir.y);
    if (len == 0.0f) return;
    cv::Point2f ud = dir / len;

    float radius = thickness * 0.5f;
    float blendW = antialias ? 1.0f : 0.0f;

    std::vector<cv::Point> ends = { p0, p1 };
    cv::Rect bbox = cv::boundingRect(ends);
    bbox.x     -= thickness; bbox.y -= thickness;
    bbox.width  += 2*thickness;
    bbox.height += 2*thickness;

    for (int y = bbox.y; y < bbox.y + bbox.height; ++y) {
        for (int x = bbox.x; x < bbox.x + bbox.width; ++x) {
            cv::Point2f pt(x + 0.5f, y + 0.5f);
            cv::Point2f v = pt - cv::Point2f(p0);
            float t = v.dot(ud);
            t = std::clamp(t, 0.0f, len);
            cv::Point2f proj = cv::Point2f(p0) + ud * t;
            float d = cv::norm(pt - proj);

            float alpha;
            if (d <= radius) {
                alpha = 1.0f;
            }
            else if (antialias && d <= radius + blendW) {
                alpha = 1.0f - smoothstep(radius, radius + blendW, d);
            }
            else {
                alpha = 0.0f;
            }

            if (alpha >= 1.0f) {
                img.at<cv::Vec3b>(y, x) = color;
            }
            else if (alpha > 0.0f) {
                plot(img, x, y, alpha, color);
            }
        }
    }
}

void drawLineDDA(cv::Mat &img, cv::Point p0, cv::Point p1,
                 const cv::Vec3b &color, int thickness)
{
    if (use_antialiasing && thickness == 1) {
        drawLineGuptaSproull(img, p0, p1, color, float(thickness));
    }
    else if (thickness > 1) {
        drawLineThickAA(img, p0, p1, color, thickness, use_antialiasing);
    }
    else {
        float dx = float(p1.x - p0.x);
        float dy = float(p1.y - p0.y);
        int steps = std::max(abs(p1.x - p0.x), abs(p1.y - p0.y));
        if (steps == 0) {
            setPixel(img, p0.x, p0.y, color);
            return;
        }
        float xInc = dx / steps, yInc = dy / steps;
        float x = float(p0.x), y = float(p0.y);
        for (int i = 0; i <= steps; ++i) {
            setPixel(img, int(std::round(x)), int(std::round(y)), color);
            x += xInc; y += yInc;
        }
    }
}

//————————————————————————————————————————————————————————————————————
// PROJECT 3D → 2D (DROP Z + ROUND)

static inline cv::Point projectVertex(const cv::Point3f &v3) {
    return cv::Point(
        int(std::round(v3.x)),
        int(std::round(v3.y))
    );
}

//————————————————————————————————————————————————————————————————————
// DRAW CUBE MESH

void drawCubeMesh(cv::Mat &img, const Cube &cube) {
    int totalVerts = int(cube.vertices.size());  // should be 8
    vector<cv::Point> proj2d(totalVerts);
    for (int i = 0; i < totalVerts; ++i) {
        proj2d[i] = projectVertex(cube.vertices[i]);
    }
    for (auto &f : cube.faces) {
        int i0 = f[0], i1 = f[1], i2 = f[2];
        drawLineDDA(img, proj2d[i0], proj2d[i1], cube.color, 1);
        drawLineDDA(img, proj2d[i1], proj2d[i2], cube.color, 1);
        drawLineDDA(img, proj2d[i2], proj2d[i0], cube.color, 1);
    }
}

//————————————————————————————————————————————————————————————————————
// DRAW CYLINDER MESH

void drawCylinderMesh(cv::Mat &img, const Cylinder &cyl) {
    int totalVerts = int(cyl.vertices.size());
    vector<cv::Point> proj2d(totalVerts);
    for (int i = 0; i < totalVerts; ++i) {
        proj2d[i] = projectVertex(cyl.vertices[i]);
    }
    for (auto &f : cyl.faces) {
        int i0 = f[0], i1 = f[1], i2 = f[2];
        drawLineDDA(img, proj2d[i0], proj2d[i1], cyl.color, 1);
        drawLineDDA(img, proj2d[i1], proj2d[i2], cyl.color, 1);
        drawLineDDA(img, proj2d[i2], proj2d[i0], cyl.color, 1);
    }
}

//————————————————————————————————————————————————————————————————————
// DRAW SPHERE MESH

void drawSphereMesh(cv::Mat &img, const Sphere &sph) {
    int totalVerts = int(sph.vertices.size());
    vector<cv::Point> proj2d(totalVerts);
    for (int i = 0; i < totalVerts; ++i) {
        proj2d[i] = projectVertex(sph.vertices[i]);
    }
    for (auto &f : sph.faces) {
        int i0 = f[0], i1 = f[1], i2 = f[2];
        drawLineDDA(img, proj2d[i0], proj2d[i1], sph.color, 1);
        drawLineDDA(img, proj2d[i1], proj2d[i2], sph.color, 1);
        drawLineDDA(img, proj2d[i2], proj2d[i0], sph.color, 1);
    }
}

//————————————————————————————————————————————————————————————————————
// DRAW CONE MESH

void drawConeMesh(cv::Mat &img, const Cone &cone) {
    int totalVerts = int(cone.vertices.size());
    vector<cv::Point> proj2d(totalVerts);
    for (int i = 0; i < totalVerts; ++i) {
        proj2d[i] = projectVertex(cone.vertices[i]);
    }
    for (auto &f : cone.faces) {
        int i0 = f[0], i1 = f[1], i2 = f[2];
        drawLineDDA(img, proj2d[i0], proj2d[i1], cone.color, 1);
        drawLineDDA(img, proj2d[i1], proj2d[i2], cone.color, 1);
        drawLineDDA(img, proj2d[i2], proj2d[i0], cone.color, 1);
    }
}

//————————————————————————————————————————————————————————————————————
// MOUSE & SCROLL CALLBACKS

double extract_scroll_delta(GdkEventScroll *event) {
    double delta = 0;
    if (event->direction == GDK_SCROLL_SMOOTH) {
        gdouble dx = 0, dy = 0;
        gdk_event_get_scroll_deltas(reinterpret_cast<GdkEvent *>(event), &dx, &dy);
        delta = dy;
    }
    else if (event->direction == GDK_SCROLL_UP) {
        delta = -1;
    }
    else if (event->direction == GDK_SCROLL_DOWN) {
        delta = +1;
    }
    return delta;
}

gboolean on_mouse_click(GtkWidget *widget, GdkEventButton *event, gpointer) {
    cv::Point pt = map_click_to_image(widget, event->x, event->y);

    // Ensure the drawing_area has focus so we can detect Shift correctly:
    gtk_widget_grab_focus(drawing_area);

    // MIDDLE click = select topmost shape under cursor
    if (event->button == GDK_BUTTON_MIDDLE) {
        selectedShapeIndex = -1;

        // 1) Check cubes (drawn “behind” but we choose an order here):
        for (int i = int(cubes.size()) - 1; i >= 0; --i) {
            vector<cv::Point> proj;
            proj.reserve(cubes[i].vertices.size());
            for (auto &v : cubes[i].vertices) {
                proj.emplace_back(
                    int(std::round(v.x)),
                    int(std::round(v.y))
                );
            }
            cv::Rect bbox = cv::boundingRect(proj);
            if (bbox.contains(pt)) {
                selectedShapeIndex = i;
                break;
            }
        }
        if (selectedShapeIndex >= 0) {
            gtk_widget_queue_draw(widget);
            return TRUE;
        }

        // 2) Check cylinders (index offset by cubes.size())
        for (int i = int(cylinders.size()) - 1; i >= 0; --i) {
            vector<cv::Point> proj;
            proj.reserve(cylinders[i].vertices.size());
            for (auto &v : cylinders[i].vertices) {
                proj.emplace_back(
                    int(std::round(v.x)),
                    int(std::round(v.y))
                );
            }
            cv::Rect bbox = cv::boundingRect(proj);
            if (bbox.contains(pt)) {
                selectedShapeIndex = int(cubes.size()) + i;
                break;
            }
        }
        if (selectedShapeIndex >= 0) {
            gtk_widget_queue_draw(widget);
            return TRUE;
        }

        // 3) Check spheres (index offset by cubes.size()+cylinders.size())
        for (int i = int(spheres.size()) - 1; i >= 0; --i) {
            vector<cv::Point> proj;
            proj.reserve(spheres[i].vertices.size());
            for (auto &v : spheres[i].vertices) {
                proj.emplace_back(
                    int(std::round(v.x)),
                    int(std::round(v.y))
                );
            }
            cv::Rect bbox = cv::boundingRect(proj);
            if (bbox.contains(pt)) {
                selectedShapeIndex = int(cubes.size() + cylinders.size()) + i;
                break;
            }
        }
        if (selectedShapeIndex >= 0) {
            gtk_widget_queue_draw(widget);
            return TRUE;
        }

        // 4) Check cones (index offset by cubes.size()+cylinders.size()+spheres.size())
        for (int i = int(cones.size()) - 1; i >= 0; --i) {
            vector<cv::Point> proj;
            proj.reserve(cones[i].vertices.size());
            for (auto &v : cones[i].vertices) {
                proj.emplace_back(
                    int(std::round(v.x)),
                    int(std::round(v.y))
                );
            }
            cv::Rect bbox = cv::boundingRect(proj);
            if (bbox.contains(pt)) {
                selectedShapeIndex = int(cubes.size() + cylinders.size() + spheres.size()) + i;
                break;
            }
        }
        gtk_widget_queue_draw(widget);
        return TRUE;
    }

    // LEFT click = place a new shape at (pt.x, pt.y, 0)
    if (event->button == GDK_BUTTON_PRIMARY) {
        if (currentShape == ShapeType::Cube) {
            Cube c({ float(pt.x), float(pt.y), 0.0f }, DEFAULT_CUBE_SIZE);
            cubes.push_back(c);
        }
        else if (currentShape == ShapeType::Cylinder) {
            Cylinder c({ float(pt.x), float(pt.y), 0.0f },
                       DEFAULT_CYL_RADIUS,
                       DEFAULT_CYL_HEIGHT,
                       DEFAULT_CYL_SUBDIV);
            cylinders.push_back(c);
        }
        else if (currentShape == ShapeType::Sphere) {
            Sphere s({ float(pt.x), float(pt.y), 0.0f },
                     DEFAULT_SPH_RADIUS,
                     DEFAULT_SPH_LAT_BANDS,
                     DEFAULT_SPH_LON_BANDS);
            spheres.push_back(s);
        }
        else if (currentShape == ShapeType::Cone) {
            Cone c({ float(pt.x), float(pt.y), 0.0f },
                   DEFAULT_CONE_RADIUS,
                   DEFAULT_CONE_HEIGHT,
                   DEFAULT_CONE_SUBDIV);
            cones.push_back(c);
        }
        gtk_widget_queue_draw(widget);
        return TRUE;
    }

    return FALSE;
}

gboolean on_scroll(GtkWidget *widget, GdkEventScroll *event, gpointer) {
    double delta = extract_scroll_delta(event);
    if (delta == 0.0) return FALSE;
    float wheelAmt = static_cast<float>(delta);

    // If no shape is selected, do nothing
    if (selectedShapeIndex < 0) {
        return FALSE;
    }

    // ------- 1) HANDLE Ctrl + Wheel: change mesh density -------
    if (event->state & GDK_CONTROL_MASK) {
        // We'll treat "wheelAmt < 0" as "increase subdivisions",
        // and "wheelAmt > 0" as "decrease subdivisions".
        bool increase = (wheelAmt < 0);

        // Clamp settings:
        const int MIN_SUBDIV = 3;    // you can pick any reasonable minimum
        const int MAX_SUBDIV = 128;  // or whatever upper bound you like

        int idx = selectedShapeIndex;

        // 1a) Cube has no subdivisions parameter, so skip it
        if (idx < int(cubes.size())) {
            // nothing to do for a pure-cube mesh (we keep it at 8 corners)
        }
        // 1b) Cylinder: adjust subdivisions
        else if (idx < int(cubes.size() + cylinders.size())) {
            int cylIdx = idx - int(cubes.size());
            Cylinder & cyl = cylinders[cylIdx];
            int oldSub = cyl.subdivisions;
            int newSub = oldSub + (increase ? +1 : -1);
            newSub = clampInt(newSub, MIN_SUBDIV, MAX_SUBDIV);
            if (newSub != oldSub) {
                cyl.subdivisions = newSub;
                cyl.buildMesh();  // rebuild with new subdivisions
            }
        }
        // 1c) Sphere: adjust both latBands & lonBands
        else if (idx < int(cubes.size() + cylinders.size() + spheres.size())) {
            int sphIdx = idx - int(cubes.size() + cylinders.size());
            Sphere & sph = spheres[sphIdx];
            int oldLat = sph.latBands;
            int oldLon = sph.lonBands;
            int newLat = oldLat + (increase ? +1 : -1);
            int newLon = oldLon + (increase ? +1 : -1);
            newLat = clampInt(newLat, MIN_SUBDIV, MAX_SUBDIV);
            newLon = clampInt(newLon, MIN_SUBDIV, MAX_SUBDIV);
            if (newLat != oldLat || newLon != oldLon) {
                sph.latBands = newLat;
                sph.lonBands = newLon;
                sph.buildMesh();  // rebuild with new lat/lon
            }
        }
        // 1d) Cone: adjust subdivisions
        else {
            int coneIdx = idx - int(cubes.size() + cylinders.size() + spheres.size());
            Cone & cone = cones[coneIdx];
            int oldSub = cone.subdivisions;
            int newSub = oldSub + (increase ? +1 : -1);
            newSub = clampInt(newSub, MIN_SUBDIV, MAX_SUBDIV);
            if (newSub != oldSub) {
                cone.subdivisions = newSub;
                cone.buildMesh();  // rebuild with new subdivisions
            }
        }

        gtk_widget_queue_draw(widget);
        return TRUE;  // we handled Ctrl+scroll
    }

    // ------- 2) SHIFT + Wheel: scale shape (as before) -------
    if (event->state & GDK_SHIFT_MASK) {
        const float k = 0.1f;
        float scaleFactor = (wheelAmt < 0 ? 1.0f + k : 1.0f - k);

        int idx = selectedShapeIndex;
        // Cube range: [0 .. cubes.size()-1]
        if (idx < int(cubes.size())) {
            cubes[idx].scale(scaleFactor);
        }
        // Cylinder range: [cubes.size() .. cubes.size()+cylinders.size()-1]
        else if (idx < int(cubes.size() + cylinders.size())) {
            int cylIdx = idx - int(cubes.size());
            cylinders[cylIdx].scale(scaleFactor);
        }
        // Sphere range: [cubes.size()+cylinders.size() .. cubes+…+cylinders+spheres.size()-1]
        else if (idx < int(cubes.size() + cylinders.size() + spheres.size())) {
            int sphIdx = idx - int(cubes.size() + cylinders.size());
            spheres[sphIdx].scale(scaleFactor);
        }
        // Cone range: everything else
        else {
            int coneIdx = idx - int(cubes.size() + cylinders.size() + spheres.size());
            cones[coneIdx].scale(scaleFactor);
        }

        gtk_widget_queue_draw(widget);
        return TRUE;
    }

    // ------- 3) Plain Wheel: rotate shape (as before) -------
    float angle = wheelAmt * (5.0f * CV_PI / 180.0f);
    int idx = selectedShapeIndex;
    // 3a) Rotate cube
    if (idx < int(cubes.size())) {
        switch (currentAxis) {
            case Axis::X: cubes[idx].rotateX(angle); break;
            case Axis::Y: cubes[idx].rotateY(angle); break;
            case Axis::Z: cubes[idx].rotateZ(angle); break;
        }
    }
    // 3b) Rotate cylinder
    else if (idx < int(cubes.size() + cylinders.size())) {
        int cylIdx = idx - int(cubes.size());
        switch (currentAxis) {
            case Axis::X: cylinders[cylIdx].rotateX(angle); break;
            case Axis::Y: cylinders[cylIdx].rotateY(angle); break;
            case Axis::Z: cylinders[cylIdx].rotateZ(angle); break;
        }
    }
    // 3c) Rotate sphere
    else if (idx < int(cubes.size() + cylinders.size() + spheres.size())) {
        int sphIdx = idx - int(cubes.size() + cylinders.size());
        switch (currentAxis) {
            case Axis::X: spheres[sphIdx].rotateX(angle); break;
            case Axis::Y: spheres[sphIdx].rotateY(angle); break;
            case Axis::Z: spheres[sphIdx].rotateZ(angle); break;
        }
    }
    // 3d) Rotate cone
    else {
        int coneIdx = idx - int(cubes.size() + cylinders.size() + spheres.size());
        switch (currentAxis) {
            case Axis::X: cones[coneIdx].rotateX(angle); break;
            case Axis::Y: cones[coneIdx].rotateY(angle); break;
            case Axis::Z: cones[coneIdx].rotateZ(angle); break;
        }
    }

    gtk_widget_queue_draw(widget);
    return TRUE;
}


//————————————————————————————————————————————————————————————————————
// KEY PRESS/RELEASE (switch rotation axis)

gboolean on_key_press(GtkWidget*, GdkEventKey *ev, gpointer) {
    switch(ev->keyval) {
      case GDK_KEY_x: case GDK_KEY_X:
        currentAxis = Axis::X; return TRUE;
      case GDK_KEY_y: case GDK_KEY_Y:
        currentAxis = Axis::Y; return TRUE;
      case GDK_KEY_z: case GDK_KEY_Z:
        currentAxis = Axis::Z; return TRUE;
    }
    return FALSE;
}

gboolean on_key_release(GtkWidget*, GdkEventKey *ev, gpointer) {
    if (ev->keyval == GDK_KEY_x || ev->keyval == GDK_KEY_X
     || ev->keyval == GDK_KEY_y || ev->keyval == GDK_KEY_Y) 
    {
        currentAxis = Axis::Z;
        return TRUE;
    }
    return FALSE;
}

//————————————————————————————————————————————————————————————————————
// GTK DRAW CALLBACK: CLEAR → DRAW ALL SHAPES → BLIT

gboolean draw_callback(GtkWidget *widget, cairo_t *cr, gpointer) {
    // 1) Clear to white
    image.setTo(cv::Scalar(255,255,255));

    // 2) Draw all cubes (mesh)
    for (int i = 0; i < int(cubes.size()); ++i) {
    if (selectedShapeIndex == i) {
        Cube tmp = cubes[i];
        tmp.color = {0,0,255};   // highlight the selected cube in blue
        drawCubeMesh(image, tmp);
    }
    else {
        drawCubeMesh(image, cubes[i]);
    }
}


    // 3) Draw all cylinders (mesh)
    for (int i = 0; i < int(cylinders.size()); ++i) {
        int idx = int(cubes.size()) + i;
        if (selectedShapeIndex == idx) {
            Cylinder tmp = cylinders[i];
            tmp.color = {0,0,255};
            drawCylinderMesh(image, tmp);
        } 
        else {
            drawCylinderMesh(image, cylinders[i]);
        }
    }

    // 4) Draw all spheres (mesh)
    for (int i = 0; i < int(spheres.size()); ++i) {
        int idx = int(cubes.size() + cylinders.size()) + i;
        if (selectedShapeIndex == idx) {
            Sphere tmp = spheres[i];
            tmp.color = {0,0,255};
            drawSphereMesh(image, tmp);
        }
        else {
            drawSphereMesh(image, spheres[i]);
        }
    }

    // 5) Draw all cones (mesh)
    for (int i = 0; i < int(cones.size()); ++i) {
        int idx = int(cubes.size() + cylinders.size() + spheres.size()) + i;
        if (selectedShapeIndex == idx) {
            Cone tmp = cones[i];
            tmp.color = {0,0,255};
            drawConeMesh(image, tmp);
        }
        else {
            drawConeMesh(image, cones[i]);
        }
    }

    // 6) Push image to GTK
    cv::Mat rgb_image;
    cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);

    GdkPixbuf *pixbuf = gdk_pixbuf_new_from_data(
        rgb_image.data, GDK_COLORSPACE_RGB, FALSE, 8,
        rgb_image.cols, rgb_image.rows, rgb_image.step,
        NULL, NULL
    );

    int w = gtk_widget_get_allocated_width(widget);
    int h = gtk_widget_get_allocated_height(widget);
    double sx = double(w) / image.cols;
    double sy = double(h) / image.rows;
    current_scale = std::min(sx, sy);

    int draw_w = int(image.cols * current_scale);
    int draw_h = int(image.rows * current_scale);
    int ox = (w - draw_w) / 2;
    int oy = (h - draw_h) / 2;

    cairo_translate(cr, ox, oy);
    cairo_scale(cr, current_scale, current_scale);
    gdk_cairo_set_source_pixbuf(cr, pixbuf, 0, 0);
    cairo_paint(cr);

    g_object_unref(pixbuf);
    return FALSE;
}

//————————————————————————————————————————————————————————————————————
// GTK MENU CALLBACKS FOR SHAPE SELECTION & ANTIALIASING

static void on_cube_activate(GtkMenuItem*, gpointer) {
    currentShape = ShapeType::Cube;
}

static void on_cylinder_activate(GtkMenuItem*, gpointer) {
    currentShape = ShapeType::Cylinder;
}

static void on_sphere_activate(GtkMenuItem*, gpointer) {
    currentShape = ShapeType::Sphere;
}

static void on_cone_activate(GtkMenuItem*, gpointer) {
    currentShape = ShapeType::Cone;
}

static void on_aliasing_toggle(GtkMenuItem*, gpointer) {
    use_antialiasing = !use_antialiasing;
    gtk_widget_queue_draw(drawing_area);
}

//————————————————————————————————————————————————————————————————————
// GTK MENU CALLBACKS FOR SCENE LOAD/SAVE

static void on_load_scene_activate(GtkMenuItem*, gpointer) {
    GtkWidget *dialog = gtk_file_chooser_dialog_new(
        "Load Scene File",
        GTK_WINDOW(gtk_widget_get_toplevel(drawing_area)),
        GTK_FILE_CHOOSER_ACTION_OPEN,
        "_Cancel", GTK_RESPONSE_CANCEL,
        "_Open",   GTK_RESPONSE_ACCEPT,
        NULL
    );
    GtkFileFilter *filter = gtk_file_filter_new();
    gtk_file_filter_set_name(filter, "Scene Files (*.scene)");
    gtk_file_filter_add_pattern(filter, "*.scene");
    gtk_file_chooser_add_filter(GTK_FILE_CHOOSER(dialog), filter);

    if (gtk_dialog_run(GTK_DIALOG(dialog)) == GTK_RESPONSE_ACCEPT) {
        char *filename_c = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dialog));
        std::string filename(filename_c);
        g_free(filename_c);

        cubes.clear();
        cylinders.clear();
        spheres.clear();
        cones.clear();
        selectedShapeIndex = -1;
        if (!loadScene(filename)) {
            std::cerr << "Failed to load scene from " << filename << "\n";
        }
        gtk_widget_queue_draw(drawing_area);
    }
    gtk_widget_destroy(dialog);
}

static void on_save_scene_activate(GtkMenuItem*, gpointer) {
    GtkWidget *dialog = gtk_file_chooser_dialog_new(
        "Save Scene File",
        GTK_WINDOW(gtk_widget_get_toplevel(drawing_area)),
        GTK_FILE_CHOOSER_ACTION_SAVE,
        "_Cancel", GTK_RESPONSE_CANCEL,
        "_Save",   GTK_RESPONSE_ACCEPT,
        NULL
    );
    gtk_file_chooser_set_do_overwrite_confirmation(GTK_FILE_CHOOSER(dialog), TRUE);

    GtkFileFilter *filter = gtk_file_filter_new();
    gtk_file_filter_set_name(filter, "Scene Files (*.scene)");
    gtk_file_filter_add_pattern(filter, "*.scene");
    gtk_file_chooser_add_filter(GTK_FILE_CHOOSER(dialog), filter);

    if (gtk_dialog_run(GTK_DIALOG(dialog)) == GTK_RESPONSE_ACCEPT) {
        char *filename_c = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dialog));
        std::string filename(filename_c);
        g_free(filename_c);

        if (filename.size() < 6 || filename.substr(filename.size() - 6) != ".scene") {
            filename += ".scene";
        }

        if (!saveScene(filename)) {
            std::cerr << "Failed to save scene to " << filename << "\n";
        }
    }
    gtk_widget_destroy(dialog);
}

//————————————————————————————————————————————————————————————————————
// GTK MENU CREATION

GtkWidget* create_main_menu() {
    GtkWidget *menubar = gtk_menu_bar_new();

    // — File Menu —
    GtkWidget *file_item = gtk_menu_item_new_with_label("File");
    gtk_menu_shell_append(GTK_MENU_SHELL(menubar), file_item);
    GtkWidget *file_menu = gtk_menu_new();
    gtk_menu_item_set_submenu(GTK_MENU_ITEM(file_item), file_menu);

    //   Scene Submenu
    GtkWidget *scene_item = gtk_menu_item_new_with_label("Scene");
    gtk_menu_shell_append(GTK_MENU_SHELL(file_menu), scene_item);
    GtkWidget *scene_menu = gtk_menu_new();
    gtk_menu_item_set_submenu(GTK_MENU_ITEM(scene_item), scene_menu);

    GtkWidget *load_item = gtk_menu_item_new_with_label("Load Scene…");
    GtkWidget *save_item = gtk_menu_item_new_with_label("Save Scene…");
    gtk_menu_shell_append(GTK_MENU_SHELL(scene_menu), load_item);
    gtk_menu_shell_append(GTK_MENU_SHELL(scene_menu), save_item);

    g_signal_connect(load_item, "activate",
                     G_CALLBACK(on_load_scene_activate), nullptr);
    g_signal_connect(save_item, "activate",
                     G_CALLBACK(on_save_scene_activate), nullptr);

    //   Toggle Antialiasing
    GtkWidget *alias_item = gtk_menu_item_new_with_label("Toggle Antialiasing");
    gtk_menu_shell_append(GTK_MENU_SHELL(file_menu), alias_item);
    g_signal_connect(alias_item, "activate",
                     G_CALLBACK(on_aliasing_toggle), nullptr);

    // — 3D Shapes Menu —
    GtkWidget *shapes_item = gtk_menu_item_new_with_label("3D Shapes");
    gtk_menu_shell_append(GTK_MENU_SHELL(menubar), shapes_item);
    GtkWidget *shapes_menu = gtk_menu_new();
    gtk_menu_item_set_submenu(GTK_MENU_ITEM(shapes_item), shapes_menu);

    GtkWidget *cube_item     = gtk_menu_item_new_with_label("Cube");
    GtkWidget *cylinder_item = gtk_menu_item_new_with_label("Cylinder");
    GtkWidget *sphere_item   = gtk_menu_item_new_with_label("Sphere");
    GtkWidget *cone_item     = gtk_menu_item_new_with_label("Cone");
    gtk_menu_shell_append(GTK_MENU_SHELL(shapes_menu), cube_item);
    gtk_menu_shell_append(GTK_MENU_SHELL(shapes_menu), cylinder_item);
    gtk_menu_shell_append(GTK_MENU_SHELL(shapes_menu), sphere_item);
    gtk_menu_shell_append(GTK_MENU_SHELL(shapes_menu), cone_item);

    g_signal_connect(cube_item,     "activate",
                     G_CALLBACK(on_cube_activate),     nullptr);
    g_signal_connect(cylinder_item, "activate",
                     G_CALLBACK(on_cylinder_activate), nullptr);
    g_signal_connect(sphere_item,   "activate",
                     G_CALLBACK(on_sphere_activate),   nullptr);
    g_signal_connect(cone_item,     "activate",
                     G_CALLBACK(on_cone_activate),     nullptr);

    return menubar;
}

//————————————————————————————————————————————————————————————————————
// SCENE LOADER & SAVER (supports cubes, cylinders, spheres, cones)

bool loadScene(const std::string& filename) {
    std::ifstream in(filename);
    if (!in.is_open()) {
        std::cerr << "Error: could not open scene file \"" << filename << "\"\n";
        return false;
    }

    string line;
    int lineNo = 0;
    while (std::getline(in, line)) {
        ++lineNo;
        auto firstNonSpace = line.find_first_not_of(" \t\r\n");
        if (firstNonSpace == string::npos) continue;    // blank
        if (line[firstNonSpace] == '#') continue;       // comment

        std::istringstream iss(line);
        string token;
        iss >> token;
        if (token == "cube") {
            float cx, cy, cz, side;
            if (!(iss >> cx >> cy >> cz >> side)) {
                std::cerr << "Parse error at line " << lineNo
                          << ": expected `cube cx cy cz side`\n";
                continue;
            }
            float m[9];
            bool hasMat = true;
            for (int i = 0; i < 9; ++i) {
                if (!(iss >> m[i])) { hasMat = false; break; }
            }
            cv::Matx33f R = cv::Matx33f::eye();
            if (hasMat) {
                R = cv::Matx33f(
                    m[0], m[1], m[2],
                    m[3], m[4], m[5],
                    m[6], m[7], m[8]
                );
            }
            Cube c({cx, cy, cz}, side, R);
            cubes.push_back(c);
        }
        else if (token == "cylinder") {
            float cx, cy, cz, rad, h; 
            int sub;
            if (!(iss >> cx >> cy >> cz >> rad >> h >> sub)) {
                std::cerr << "Parse error at line " << lineNo
                          << ": expected `cylinder cx cy cz radius height subdivisions`\n";
                continue;
            }
            float m[9];
            bool hasMat = true;
            for (int i = 0; i < 9; ++i) {
                if (!(iss >> m[i])) { hasMat = false; break; }
            }
            cv::Matx33f R = cv::Matx33f::eye();
            if (hasMat) {
                R = cv::Matx33f(
                    m[0], m[1], m[2],
                    m[3], m[4], m[5],
                    m[6], m[7], m[8]
                );
            }
            Cylinder cyl({cx, cy, cz}, rad, h, sub, R);
            cylinders.push_back(cyl);
        }
        else if (token == "sphere") {
            float cx, cy, cz, rad; 
            int latB, lonB;
            if (!(iss >> cx >> cy >> cz >> rad >> latB >> lonB)) {
                std::cerr << "Parse error at line " << lineNo
                          << ": expected `sphere cx cy cz radius latBands lonBands`\n";
                continue;
            }
            float m[9];
            bool hasMat = true;
            for (int i = 0; i < 9; ++i) {
                if (!(iss >> m[i])) { hasMat = false; break; }
            }
            cv::Matx33f R = cv::Matx33f::eye();
            if (hasMat) {
                R = cv::Matx33f(
                    m[0], m[1], m[2],
                    m[3], m[4], m[5],
                    m[6], m[7], m[8]
                );
            }
            Sphere s({cx, cy, cz}, rad, latB, lonB, R);
            spheres.push_back(s);
        }
        else if (token == "cone") {
            float cx, cy, cz, rad, h; 
            int sub;
            if (!(iss >> cx >> cy >> cz >> rad >> h >> sub)) {
                std::cerr << "Parse error at line " << lineNo
                          << ": expected `cone cx cy cz radius height subdivisions`\n";
                continue;
            }
            float m[9];
            bool hasMat = true;
            for (int i = 0; i < 9; ++i) {
                if (!(iss >> m[i])) { hasMat = false; break; }
            }
            cv::Matx33f R = cv::Matx33f::eye();
            if (hasMat) {
                R = cv::Matx33f(
                    m[0], m[1], m[2],
                    m[3], m[4], m[5],
                    m[6], m[7], m[8]
                );
            }
            Cone c({cx, cy, cz}, rad, h, sub, R);
            cones.push_back(c);
        }
        else {
            std::cerr << "Warning: unrecognized token \"" << token
                      << "\" on line " << lineNo << "\n";
        }
    }

    in.close();
    std::cout << "Loaded "
              << cubes.size() << " cube(s), "
              << cylinders.size() << " cylinder(s), "
              << spheres.size() << " sphere(s), "
              << cones.size() << " cone(s) from \""
              << filename << "\".\n";
    return true;
}

bool saveScene(const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Error: could not open file for writing \"" << filename << "\"\n";
        return false;
    }

    // Write cubes first
    for (const auto &c : cubes) {
        // "cube cx cy cz side"
        out << "cube "
            << c.center.x << " "
            << c.center.y << " "
            << c.center.z << " "
            << c.sideLength;
        const cv::Matx33f &R = c.orientation;
        out << "  "
            << R(0,0) << " " << R(0,1) << " " << R(0,2) << "  "
            << R(1,0) << " " << R(1,1) << " " << R(1,2) << "  "
            << R(2,0) << " " << R(2,1) << " " << R(2,2) << "\n";
    }

    // Then write cylinders
    for (const auto &c : cylinders) {
        // "cylinder cx cy cz radius height subdivisions"
        out << "cylinder "
            << c.center.x << " "
            << c.center.y << " "
            << c.center.z << " "
            << c.radius   << " "
            << c.height   << " "
            << c.subdivisions;
        const cv::Matx33f &R = c.orientation;
        out << "  "
            << R(0,0) << " " << R(0,1) << " " << R(0,2) << "  "
            << R(1,0) << " " << R(1,1) << " " << R(1,2) << "  "
            << R(2,0) << " " << R(2,1) << " " << R(2,2) << "\n";
    }

    // Then write spheres
    for (const auto &s : spheres) {
        // "sphere cx cy cz radius latBands lonBands"
        out << "sphere "
            << s.center.x << " "
            << s.center.y << " "
            << s.center.z << " "
            << s.radius  << " "
            << s.latBands << " "
            << s.lonBands;
        const cv::Matx33f &R = s.orientation;
        out << "  "
            << R(0,0) << " " << R(0,1) << " " << R(0,2) << "  "
            << R(1,0) << " " << R(1,1) << " " << R(1,2) << "  "
            << R(2,0) << " " << R(2,1) << " " << R(2,2) << "\n";
    }

    // Finally write cones
    for (const auto &c : cones) {
        // "cone cx cy cz radius height subdivisions"
        out << "cone "
            << c.center.x << " "
            << c.center.y << " "
            << c.center.z << " "
            << c.radius  << " "
            << c.height  << " "
            << c.subdivisions;
        const cv::Matx33f &R = c.orientation;
        out << "  "
            << R(0,0) << " " << R(0,1) << " " << R(0,2) << "  "
            << R(1,0) << " " << R(1,1) << " " << R(1,2) << "  "
            << R(2,0) << " " << R(2,1) << " " << R(2,2) << "\n";
    }

    out.close();
    std::cout << "Saved "
              << cubes.size() << " cube(s), "
              << cylinders.size() << " cylinder(s), "
              << spheres.size() << " sphere(s), "
              << cones.size() << " cone(s) to \""
              << filename << "\".\n";
    return true;
}

//————————————————————————————————————————————————————————————————————
// MAIN: INITIALIZE, SET UP GTK, AND RUN

int main(int argc, char *argv[]) {
    // 1) If a scene filename is provided, load it first:
    if (argc > 1) {
        std::string sceneFile = argv[1];
        if (!loadScene(sceneFile)) {
            return 1;
        }
    }

    // 2) Create a blank white image to draw on
    image = cv::Mat::zeros(600, 800, CV_8UC3);
    image.setTo(cv::Scalar(255,255,255));

    // 3) Initialize GTK
    gtk_init(&argc, &argv);

    GtkWidget *window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_title(GTK_WINDOW(window),
                        "3D Scene: Cubes, Cylinders, Spheres, Cones");
    gtk_window_set_default_size(GTK_WINDOW(window), 1000, 800);

    GtkWidget *vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    gtk_container_add(GTK_CONTAINER(window), vbox);

    // 4) Add the menu bar
    GtkWidget *menuBar = create_main_menu();
    gtk_box_pack_start(GTK_BOX(vbox), menuBar, FALSE, FALSE, 0);

    // 5) Create a drawing area
    drawing_area = gtk_drawing_area_new();
    gtk_widget_set_hexpand(drawing_area, TRUE);
    gtk_widget_set_vexpand(drawing_area, TRUE);
    gtk_widget_set_can_focus(drawing_area, TRUE); // allow focus for Shift detection
    gtk_box_pack_start(GTK_BOX(vbox), drawing_area, TRUE, TRUE, 0);

    // 6) Set up event masks and signals
    gtk_widget_add_events(drawing_area,
        GDK_BUTTON_PRESS_MASK |
        GDK_SCROLL_MASK      |
        GDK_SMOOTH_SCROLL_MASK
    );
    g_signal_connect(drawing_area, "draw",
                     G_CALLBACK(draw_callback), NULL);
    g_signal_connect(drawing_area, "button-press-event",
                     G_CALLBACK(on_mouse_click), NULL);
    g_signal_connect(drawing_area, "scroll-event",
                     G_CALLBACK(on_scroll), NULL);
    g_signal_connect(window, "destroy",
                     G_CALLBACK(gtk_main_quit), NULL);

    gtk_widget_set_can_focus(window, TRUE);
    g_signal_connect(window, "key-press-event",
                     G_CALLBACK(on_key_press), NULL);
    g_signal_connect(window, "key-release-event",
                     G_CALLBACK(on_key_release), NULL);

    gtk_widget_show_all(window);
    gtk_main();
    return 0;
}
