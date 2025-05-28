#include <cmath>
#include <fstream>
#include <gtk/gtk.h>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;

enum class ShapeType { Cube, Tetrahedron};
ShapeType currentShape = ShapeType::Cube;
GtkWidget *shape_menu;

constexpr float DEFAULT_CUBE_SIZE = 50.0f;
int selectedCube = -1;

enum class Axis { X, Y, Z };
Axis currentAxis = Axis::Z;

struct Line {
  cv::Point start, end;
  int thickness;
  cv::Vec3b color = {0, 0, 255}; // default red
};

struct Cube {
    cv::Point3f      center;
    float            sideLength;
    cv::Vec3b        color     = {0, 255, 0};
    std::vector<cv::Point3f> vertices;

    // NEW: store current orientation
    cv::Matx33f      orientation = cv::Matx33f::eye();

    Cube(const cv::Point3f& c, float s, const cv::Vec3b& col = {0,255,0})
      : center(c), sideLength(s), color(col)
    {
        computeVertices();
    }

    // apply rotation about local Z
    void rotateZ(float angleRadians) {
        float c = std::cos(angleRadians), s = std::sin(angleRadians);
        cv::Matx33f Rz(
           c, -s, 0,
           s,  c, 0,
           0,  0, 1
        );
        orientation = Rz * orientation;
        computeVertices();
    }
     void rotateX(float angle) {
        float c = std::cos(angle), s = std::sin(angle);
        cv::Matx33f Rx(
            1,  0, 0,
            0,  c,-s,
            0,  s, c
        );
        orientation = Rx * orientation;
        computeVertices();
    }

    // rotate about local Y
    void rotateY(float angle) {
        float c = std::cos(angle), s = std::sin(angle);
        cv::Matx33f Ry(
             c, 0, s,
             0, 1, 0,
            -s, 0, c
        );
        orientation = Ry * orientation;
        computeVertices();
    }

    // rebuild the 8 corner points in world coords
    void computeVertices() {
        float h = sideLength * 0.5f;
        // base corners in local (center = origin)
        static const cv::Point3f base[8] = {
            {-h,-h,-h},{+h,-h,-h},{+h,+h,-h},{-h,+h,-h},
            {-h,-h,+h},{+h,-h,+h},{+h,+h,+h},{-h,+h,+h}
        };

        vertices.clear();
        for (int i = 0; i < 8; ++i) {
            cv::Vec3f v(base[i].x, base[i].y, base[i].z);
            cv::Vec3f vr = orientation * v;
            vertices.emplace_back(
              center.x + vr[0],
              center.y + vr[1],
              center.z + vr[2]
            );
        }
    }
};
struct Tetrahedron {
  std::vector<cv::Point3f> vertices;
  cv::Vec3b color = {255, 0, 0}; // default blue
};


vector<Cube> cubes;
vector<Tetrahedron> tetrahedrons;

cv::Mat image;
GtkWidget *drawing_area;
cv::Point tempPoint;
bool awaitingSecondClick = false;
double current_scale = 1.0;
bool use_antialiasing = false;


cv::Point map_click_to_image(GtkWidget *widget, double click_x,
                             double click_y) {
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

  return {std::clamp(static_cast<int>(img_x), 0, image.cols - 1),
          std::clamp(static_cast<int>(img_y), 0, image.rows - 1)};
}

double distance_to_line(cv::Point p, cv::Point a, cv::Point b) {
  cv::Point ab = b - a;
  double ab_len_sq = ab.dot(ab);

  if (ab_len_sq == 0.0)
    return cv::norm(p - a);

  double t = std::clamp((p - a).dot(ab) / ab_len_sq, 0.0, 1.0);
  cv::Point proj = a + t * ab;

  return cv::norm(p - proj);
}

float coverage(float thickness, float distance) {
  float radius = thickness / 2.0f;
  float t = distance / radius;
  return std::exp(-t * t * 0.75f);
}

void plot(cv::Mat &img, int x, int y, float alpha, const cv::Vec3b &color) {
  if (x >= 0 && x < img.cols && y >= 0 && y < img.rows) {
    cv::Vec3b &pixel = img.at<cv::Vec3b>(y, x);
    for (int i = 0; i < 3; ++i) {
      int blended =
          static_cast<int>(pixel[i] * (1.0f - alpha) + color[i] * alpha + 0.5f);
      pixel[i] = static_cast<uchar>(std::clamp(blended, 0, 255));
    }
  }
}
void drawLineGuptaSproull(cv::Mat &img, cv::Point p0, cv::Point p1,
                          const cv::Vec3b &color, float thickness = 1.0f) {
  auto plotAA = [&](int x, int y, float d) {
    float alpha = coverage(thickness, d);
    if (alpha > 0.0f)
      plot(img, x, y, alpha, color);
  };

  bool steep = std::abs(p1.y - p0.y) > std::abs(p1.x - p0.x);
  if (steep) {
    std::swap(p0.x, p0.y);
    std::swap(p1.x, p1.y);
  }
  if (p0.x > p1.x)
    std::swap(p0, p1);

  int dx = p1.x - p0.x;
  int dy = p1.y - p0.y;

  int ystep = (p1.y > p0.y) ? 1 : -1;
  float gradient = (dx == 0) ? 0.0f : static_cast<float>(dy) / dx;
  float intery = static_cast<float>(p0.y) + gradient;

  float radius = thickness / 2.0f;
  int coverageRange = std::ceil(radius) + 1;

  int x_start = p0.x + 1;
  int x_end = p1.x - 1;

  for (int x = x_start; x <= x_end; ++x) {
    int y_center = static_cast<int>(intery);
    float frac = intery - y_center;

    for (int k = -coverageRange; k <= coverageRange; ++k) {
      float d = std::abs(k - frac);
      if (steep)
        plotAA(y_center + k, x, d);
      else
        plotAA(x, y_center + k, d);
    }

    intery += gradient;
  }
}

void setPixel(cv::Mat &img, int x, int y, const cv::Vec3b &color) {
  if (x >= 0 && x < img.cols && y >= 0 && y < img.rows) {
    img.at<cv::Vec3b>(y, x) = color;
  }
}

cv::Point2f perpendicularOffset(cv::Point p0, cv::Point p1, float offset) {
  float dx = p1.x - p0.x;
  float dy = p1.y - p0.y;
  float length = std::sqrt(dx * dx + dy * dy);
  if (length == 0)
    return {0, 0};
  return {-dy * offset / length, dx * offset / length};
}

inline float smoothstep(float edge0, float edge1, float x) {
  float t = std::clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
  return t * t * (3.0f - 2.0f * t);
}

void drawLineThickPixelCopyAA(cv::Mat &img, cv::Point p0, cv::Point p1,
                              const cv::Vec3b &color, int thickness = 1,
                              bool applyAA = false) {
  cv::Point2f dir = cv::Point2f(p1) - cv::Point2f(p0);
  float len = std::hypot(dir.x, dir.y);
  if (len == 0.0f)
      return;
  cv::Point2f ud = dir / len;

  float radius     = thickness * 0.5f;
  float blendWidth = applyAA ? 1.0f : 0.0f;

  std::vector<cv::Point>  endpts = { p0, p1 };
  cv::Rect bbox = cv::boundingRect(endpts);
  bbox.x     -= thickness;
  bbox.y     -= thickness;
  bbox.width += 2 * thickness;
  bbox.height+= 2 * thickness;

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
          else if (applyAA && d <= radius + blendWidth) {

              alpha = 1.0f - smoothstep(radius, radius + blendWidth, d);
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
                 const cv::Vec3b &color, int thickness = 1) {
  if (use_antialiasing && thickness == 1) {
    drawLineGuptaSproull(img, p0, p1, color, thickness);
  } else if (thickness > 1) {
    drawLineThickPixelCopyAA(img, p0, p1, color, thickness, use_antialiasing);
  } else {
    float dx = p1.x - p0.x;
    float dy = p1.y - p0.y;
    int steps = std::max(std::abs(dx), std::abs(dy));

    float xInc = dx / steps;
    float yInc = dy / steps;

    float x = p0.x;
    float y = p0.y;

    for (int i = 0; i <= steps; ++i) {
      setPixel(img, std::round(x), std::round(y), color);
      x += xInc;
      y += yInc;
    }
  }
}

void drawCube(cv::Mat &img, const Cube &cube) {
    // Project 3D→2D (x,y) and collect into ints
    std::vector<cv::Point> proj;
    proj.reserve(8);
    for (auto &v3 : cube.vertices)
        proj.emplace_back( int(v3.x), int(v3.y) );

    // Indices of the bottom face:
    const int B[4] = {0,1,2,3};
    // Top face is just bottom+4:
    const int T[4] = {4,5,6,7};

    // draw bottom square
    for (int i = 0; i < 4; ++i) {
        drawLineDDA(img, proj[B[i]], proj[B[(i+1)%4]], cube.color, 1);
    }
    // draw top square
    for (int i = 0; i < 4; ++i) {
        drawLineDDA(img, proj[T[i]], proj[T[(i+1)%4]], cube.color, 1);
    }
    // draw the vertical edges
    for (int i = 0; i < 4; ++i) {
        drawLineDDA(img, proj[B[i]], proj[T[i]], cube.color, 1);
    }
}


double extract_scroll_delta(GdkEventScroll *event) {
  double delta = 0;
  if (event->direction == GDK_SCROLL_SMOOTH) {
    gdouble dx = 0, dy = 0;
    gdk_event_get_scroll_deltas(reinterpret_cast<GdkEvent *>(event), &dx, &dy);
    delta = dy;
  } else if (event->direction == GDK_SCROLL_UP) {
    delta = -1;
  } else if (event->direction == GDK_SCROLL_DOWN) {
    delta = 1;
  }
  return delta;
}

gboolean on_mouse_click(GtkWidget *widget, GdkEventButton *event, gpointer) {
    // map the GTK click → image coords
    cv::Point pt = map_click_to_image(widget, event->x, event->y);

    if (event->button == GDK_BUTTON_MIDDLE) {
        selectedCube = -1;
        // test topmost first
        for (int i = int(cubes.size()) - 1; i >= 0; --i) {
            std::vector<cv::Point> proj;
            proj.reserve(8);
            for (auto &v : cubes[i].vertices)
                proj.emplace_back(int(v.x), int(v.y));
            cv::Rect bbox = cv::boundingRect(proj);
            if (bbox.contains(pt)) {
                selectedCube = i;
                break;
            }
        }
        gtk_widget_queue_draw(widget);
        return TRUE;
    }

    // only respond to left-button presses
    if (event->button == GDK_BUTTON_PRIMARY) {
        if (currentShape == ShapeType::Cube) {
            // create a cube centered where you clicked, at z=0
            Cube c(cv::Point3f(pt.x, pt.y, 0.0f), DEFAULT_CUBE_SIZE);
            c.center     = cv::Point3f(pt.x, pt.y, 0.0f);
            c.sideLength = DEFAULT_CUBE_SIZE;
            // c.color uses its default {0,255,0}

            cubes.push_back(c);
        }
        else if (currentShape == ShapeType::Tetrahedron) {
            // …your existing logic for building tetrahedrons…
        }

        // trigger a redraw
        gtk_widget_queue_draw(widget);
    }

    return TRUE;
}
 gboolean on_scroll(GtkWidget *widget, GdkEventScroll *event, gpointer) {
     double delta = extract_scroll_delta(event);  // +1 or -1

    if (selectedCube >= 0) {
        float angle = float(delta) * (5.0f * CV_PI/180.0f);
        switch (currentAxis) {
          case Axis::X: cubes[selectedCube].rotateX(angle); break;
          case Axis::Y: cubes[selectedCube].rotateY(angle); break;
          case Axis::Z: /*fall-through*/ 
          default:       cubes[selectedCube].rotateZ(angle); break;
        }
        gtk_widget_queue_draw(widget);
        return TRUE;
    }

     return TRUE;
 }

 gboolean draw_callback(GtkWidget *widget, cairo_t *cr, gpointer) {

  // ── 1) clear the canvas to white each frame ──
  image.setTo(cv::Scalar(255,255,255));

  // ── 2) draw all cubes ──
    // draw & highlight selected
    for (int i = 0; i < (int)cubes.size(); ++i) {
        if (i == selectedCube) {
            Cube tmp = cubes[i];
            tmp.color = {0,0,255};         // highlight in blue
            drawCube(image, tmp);
        } else {
            drawCube(image, cubes[i]);
        }
    }

  // ── 3) draw all tetrahedra ──
//   for (const auto &tet : tetrahedrons) {
//     drawTetrahedron(image, tet);
//   }

  // Now convert to RGB for Cairo
  cv::Mat rgb_image;
  cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);
 
   GdkPixbuf *pixbuf = gdk_pixbuf_new_from_data(
       rgb_image.data, GDK_COLORSPACE_RGB, FALSE, 8, rgb_image.cols,
       rgb_image.rows, rgb_image.step, NULL, NULL);
 
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

// at file‐scope, before main():

static void
on_cube_activate(GtkMenuItem* item, gpointer data)
{
    currentShape = ShapeType::Cube;
    cout<< "Cube selected" << endl;
    gtk_widget_queue_draw(drawing_area);
}

static void
on_tetrahedral_activate(GtkMenuItem* item, gpointer data)
{
    currentShape = ShapeType::Tetrahedron;
    cout<< "Tetrahedron selected" << endl;
    gtk_widget_queue_draw(drawing_area);
}

static void
on_aliasing_toggle(GtkMenuItem* item, gpointer data)
{
    use_antialiasing = !use_antialiasing;
    cout << "Antialiasing " << (use_antialiasing ? "enabled" : "disabled") << endl;
    gtk_widget_queue_draw(drawing_area);
}

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
    // when X or Y is released, go back to Z
    if (ev->keyval==GDK_KEY_x||ev->keyval==GDK_KEY_X
     || ev->keyval==GDK_KEY_y||ev->keyval==GDK_KEY_Y)
    {
        currentAxis = Axis::Z;
        return TRUE;
    }
    return FALSE;
}


GtkWidget* create_shape_menu() {

    GtkWidget *menubar = gtk_menu_bar_new();

    GtkWidget *file_menu = gtk_menu_item_new_with_label("File");
    gtk_menu_shell_append(GTK_MENU_SHELL(menubar), file_menu);

    GtkWidget *file_submenu = gtk_menu_new();
    gtk_menu_item_set_submenu(GTK_MENU_ITEM(file_menu), file_submenu);

    GtkWidget *aliasing_item = gtk_menu_item_new_with_label("Toggle Antialiasing");
    gtk_menu_shell_append(GTK_MENU_SHELL(file_submenu), aliasing_item);
    g_signal_connect(aliasing_item, "activate", G_CALLBACK(on_aliasing_toggle), nullptr);


    GtkWidget *shapes_root = gtk_menu_item_new_with_label("3D Shapes");
    gtk_menu_shell_append(GTK_MENU_SHELL(menubar), shapes_root);

    GtkWidget *shapes_sub = gtk_menu_new();
    gtk_menu_item_set_submenu(GTK_MENU_ITEM(shapes_root), shapes_sub);

    GtkWidget *cube_item        = gtk_menu_item_new_with_label("Cube");
    GtkWidget *tetra_item       = gtk_menu_item_new_with_label("Tetrahedron");
    gtk_menu_shell_append(GTK_MENU_SHELL(shapes_sub), cube_item);
    gtk_menu_shell_append(GTK_MENU_SHELL(shapes_sub), tetra_item);


    g_signal_connect(cube_item,  "activate",
                     G_CALLBACK(on_cube_activate), nullptr);
    g_signal_connect(tetra_item, "activate",
                     G_CALLBACK(on_tetrahedral_activate), nullptr);

    return menubar;
}



int main(int argc, char *argv[]) {
    gtk_init(&argc, &argv);
    image = cv::Mat::zeros(600, 800, CV_8UC3);
    image.setTo(cv::Scalar(255, 255, 255));
    GtkWidget *window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_title(GTK_WINDOW(window), "3D Shape Editor");
    gtk_window_set_default_size(GTK_WINDOW(window), 1000, 800);

    GtkWidget *vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    gtk_container_add(GTK_CONTAINER(window), vbox);

    shape_menu = create_shape_menu();
    gtk_box_pack_start(GTK_BOX(vbox), shape_menu, FALSE, FALSE, 0);

    drawing_area = gtk_drawing_area_new();
    gtk_widget_set_hexpand(drawing_area, TRUE);
    gtk_widget_set_vexpand(drawing_area, TRUE);
    gtk_box_pack_start(GTK_BOX(vbox), drawing_area, TRUE, TRUE, 0);

    gtk_widget_add_events(drawing_area, GDK_BUTTON_PRESS_MASK | GDK_SCROLL_MASK |
                                            GDK_SMOOTH_SCROLL_MASK);

    g_signal_connect(drawing_area, "draw", G_CALLBACK(draw_callback), NULL);
    g_signal_connect(drawing_area, "button-press-event",
                    G_CALLBACK(on_mouse_click), NULL);
    g_signal_connect(drawing_area, "scroll-event", G_CALLBACK(on_scroll), NULL);
    g_signal_connect(window, "destroy", G_CALLBACK(gtk_main_quit), NULL);

    gtk_widget_set_can_focus(window, TRUE);
    g_signal_connect(window, "key-press-event",   G_CALLBACK(on_key_press),   NULL);    
    g_signal_connect(window, "key-release-event", G_CALLBACK(on_key_release), NULL);

    gtk_widget_show_all(window);
    gtk_main();
    return 0;
}