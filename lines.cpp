
#include <cmath>
#include <fstream>
#include <gtk/gtk.h>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

enum class ShapeType { Line, Circle, Polygon, Rectangle };
ShapeType currentShape = ShapeType::Line;
GtkWidget *shape_menu;

struct Line {
  cv::Point start, end;
  int thickness;
  cv::Vec3b color = {0, 0, 255}; // default red
};

struct Circle {
  cv::Point center;
  int radius;
  cv::Vec3b color = {0, 255, 0}; // default green
};

struct Polygon {
  std::vector<cv::Point> vertices;
  int thickness;
  cv::Vec3b color = {255, 0, 0}; // default blue
  bool filled = false;

  bool useImageFill = false;
  cv::Mat fillImage;
};

struct Bezier {
  std::vector<cv::Point> ctrl; // control points
  int thickness;
  cv::Vec3b color;
};

enum class EditMode { None, MoveStart, MoveEnd };

std::vector<Line> lines;
std::vector<Circle> circles;
std::vector<Polygon> polygons;

cv::Mat image;
GtkWidget *drawing_area;
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

std::vector<Bezier> beziers;
std::vector<cv::Point> currentBezier;
bool inBezierMode = false;

int selectedVertexIndex = -1;

std::vector<cv::Point> currentPolygonVertices;

bool use_antialiasing = false;

bool use_bezier = false;
bool end_bezier = false;

// Task 4

static cv::Point rectCorner;
static bool awaitingRectSecond = false;
static int selectedRectIndex = -1;
static int selectedRectVertexIndex = -1;
static bool awaitingRectCornerMove = false;
static bool awaitingRectEdgeMove = false;
static int selectedRectEdgeIndex = -1;

//  Clipping

static bool choose_clipping = false;
static bool choose_cliped = false;
static int clipedIndex = -1;
static int clippingndex = -1;

constexpr int CS_INSIDE = 0;
constexpr int CS_LEFT   = 1;    // 0001
constexpr int CS_RIGHT  = 2;    // 0010
constexpr int CS_BOTTOM = 4;    // 0100
constexpr int CS_TOP    = 8;    // 1000

int computeOutCode(const cv::Point &p, int xmin, int ymin, int xmax, int ymax) {
  int code = CS_INSIDE;
  if      (p.x < xmin) code |= CS_LEFT;
  else if (p.x > xmax) code |= CS_RIGHT;
  if      (p.y < ymin) code |= CS_BOTTOM;
  else if (p.y > ymax) code |= CS_TOP;
  return code;
}

struct Edge {
  int    yMax;      
  float  x;         
  float  invSlope;  
};

void textureFillPolygon(cv::Mat &img, const Polygon &poly) {

  if (poly.vertices.size() < 3 || poly.fillImage.empty()) return;

  int xMin = INT_MAX, xMax = INT_MIN;
  int yMin = INT_MAX, yMax = INT_MIN;
  for (auto &v : poly.vertices) {
      xMin = std::min(xMin, v.x);
      xMax = std::max(xMax, v.x);
      yMin = std::min(yMin, v.y);
      yMax = std::max(yMax, v.y);
  }

  yMin = std::max(yMin, 0);
  yMax = std::min(yMax, img.rows - 1);

  std::vector<std::vector<Edge>> edgeTable(yMax + 1);
  int N = poly.vertices.size();
  for (int i = 0; i < N; ++i) {
      cv::Point p1 = poly.vertices[i];
      cv::Point p2 = poly.vertices[(i + 1) % N];

      if (p1.y == p2.y) continue;

      int y0 = p1.y, y1 = p2.y;
      float x0 = p1.x, x1 = p2.x;
      if (y0 > y1) {
          std::swap(y0, y1);
          std::swap(x0, x1);
      }
      Edge e;
      e.yMax     = y1;                 
      e.x        = x0;                 
      e.invSlope = (x1 - x0) / float(y1 - y0);
      if (y0 >= 0 && y0 <= yMax)
          edgeTable[y0].push_back(e);
  }

  std::vector<Edge> active;
  int texW = poly.fillImage.cols,
      texH = poly.fillImage.rows;

  for (int y = yMin; y <= yMax; ++y) {

      for (auto &e : edgeTable[y]) active.push_back(e);

      active.erase(
          std::remove_if(active.begin(), active.end(),
              [y](const Edge &e){ return e.yMax <= y; }),
          active.end()
      );

      std::sort(active.begin(), active.end(),
                [](const Edge &a, const Edge &b){ return a.x < b.x; });

      for (size_t i = 0; i + 1 < active.size(); i += 2) {
          int xStart = std::max(int(std::ceil(active[i].x)),   0);
          int xEnd   = std::min(int(std::floor(active[i+1].x)), img.cols - 1);
          for (int x = xStart; x <= xEnd; ++x) {

              int u = ( (x - xMin) % texW + texW ) % texW;
              int v = ( (y - yMin) % texH + texH ) % texH;
              img.at<cv::Vec3b>(y, x) = poly.fillImage.at<cv::Vec3b>(v, u);
          }
      }

      for (auto &e : active) {
          e.x += e.invSlope;
      }
  }
}

bool cohenSutherlandClip(cv::Point p0, cv::Point p1,
                       int xmin, int ymin, int xmax, int ymax,
                       cv::Point &out0, cv::Point &out1)
{
  int code0 = computeOutCode(p0, xmin, ymin, xmax, ymax);
  int code1 = computeOutCode(p1, xmin, ymin, xmax, ymax);
  bool accept = false;

  while (true) {
      if ((code0 | code1) == 0) {
          // both inside
          accept = true;
          out0 = p0;  out1 = p1; 
          break;
      }
      else if (code0 & code1) {
          // trivial reject
          break;
      }
      else {
          // pick one outside endpoint
          int outcodeOut = code0 ? code0 : code1;
          double x, y;

          if (outcodeOut & CS_TOP) {
              x = p0.x + (p1.x - p0.x) * (ymax - p0.y) / double(p1.y - p0.y);
              y = ymax;
          }
          else if (outcodeOut & CS_BOTTOM) {
              x = p0.x + (p1.x - p0.x) * (ymin - p0.y) / double(p1.y - p0.y);
              y = ymin;
          }
          else if (outcodeOut & CS_RIGHT) {
              y = p0.y + (p1.y - p0.y) * (xmax - p0.x) / double(p1.x - p0.x);
              x = xmax;
          }
          else { // CS_LEFT
              y = p0.y + (p1.y - p0.y) * (xmin - p0.x) / double(p1.x - p0.x);
              x = xmin;
          }

          if (outcodeOut == code0) {
              p0 = cv::Point(std::round(x), std::round(y));
              code0 = computeOutCode(p0, xmin, ymin, xmax, ymax);
          } else {
              p1 = cv::Point(std::round(x), std::round(y));
              code1 = computeOutCode(p1, xmin, ymin, xmax, ymax);
          }
      }
  }
  return accept;
}



void drawCircleMidpoint(cv::Mat &img, cv::Point center, int radius,
                        const cv::Vec3b &color) {
  int x = 0, y = radius;
  int d = 1 - radius;

  auto drawSymmetricPoints = [&](int cx, int cy, int x, int y) {
    std::vector<cv::Point> points = {
        {cx + x, cy + y}, {cx - x, cy + y}, {cx + x, cy - y}, {cx - x, cy - y},
        {cx + y, cy + x}, {cx - y, cy + x}, {cx + y, cy - x}, {cx - y, cy - x}};
    for (auto &pt : points) {
      if (pt.x >= 0 && pt.x < img.cols && pt.y >= 0 && pt.y < img.rows)
        img.at<cv::Vec3b>(pt) = color;
    }
  };

  drawSymmetricPoints(center.x, center.y, x, y);
  while (x < y) {
    x++;
    if (d < 0)
      d += 2 * x + 1;
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

void draw_in_progress_polygon(cv::Mat &img) {
  if (currentPolygonVertices.size() < 2)
    return;
  if (currentPolygonVertices.size() == 1) {

    drawCircleMidpoint(img, currentPolygonVertices[0], 3,
                       cv::Vec3b(0, 200, 200));
    return;
  }
  for (size_t i = 0; i < currentPolygonVertices.size() - 1; ++i) {
    drawLineDDA(img, currentPolygonVertices[i], currentPolygonVertices[i + 1],
                cv::Vec3b(200, 200, 0), 1);
  }
}

struct EdgeEntry {
  int ymax;        
  float x;        
  float invSlope; 
};

void fillPolygon(cv::Mat &img,
               const std::vector<cv::Point> &vertices,
               const cv::Vec3b &fillColor)
{
  int n = vertices.size();
  if (n < 3) return;

  int height = img.rows;
  int ymin = INT_MAX, ymax = INT_MIN;
  for (auto &p : vertices) {
      ymin = std::min(ymin, p.y);
      ymax = std::max(ymax, p.y);
  }
  ymin = std::clamp(ymin, 0, height - 1);
  ymax = std::clamp(ymax, 0, height - 1);

  std::vector<std::vector<EdgeEntry>> ET(height);
  for (int i = 0; i < n; ++i) {
      cv::Point p1 = vertices[i];
      cv::Point p2 = vertices[(i + 1) % n];
      if (p1.y == p2.y) continue;      

      if (p1.y > p2.y) std::swap(p1, p2);

      int yMin = p1.y;
      int yMax = p2.y;
      float xAtYmin = p1.x;
      float invSlope = float(p2.x - p1.x) / float(p2.y - p1.y);

      if (yMin >= 0 && yMin < height)
          ET[yMin].push_back({ yMax, xAtYmin, invSlope });
  }

  std::vector<EdgeEntry> AET;
  for (int y = ymin; y <= ymax; ++y) {

      for (auto &e : ET[y]) AET.push_back(e);

      AET.erase(std::remove_if(AET.begin(), AET.end(),
                               [y](const EdgeEntry &e){ return e.ymax == y; }),
                AET.end());

      std::sort(AET.begin(), AET.end(),
                [](const EdgeEntry &a, const EdgeEntry &b){
                    return a.x < b.x;
                });

      for (int i = 0; i + 1 < (int)AET.size(); i += 2) {
          int xStart = std::ceil(AET[i].x);
          int xEnd   = std::floor(AET[i+1].x);
          xStart = std::clamp(xStart, 0, img.cols - 1);
          xEnd   = std::clamp(xEnd,   0, img.cols - 1);
          for (int x = xStart; x <= xEnd; ++x) {
              img.at<cv::Vec3b>(y, x) = fillColor;
          }
      }

      for (auto &e : AET)
          e.x += e.invSlope;
  }
}

void drawPolygon(cv::Mat &img, const Polygon &poly, const cv::Vec3b &color) {
  if (poly.vertices.size() < 2) return;

  if(poly.useImageFill) {
    textureFillPolygon(img, poly);
  } else if (poly.filled) {
    fillPolygon(img, poly.vertices, color);
  }
  
  for (size_t i = 0; i < poly.vertices.size(); ++i) {
      cv::Point p1 = poly.vertices[i];
      cv::Point p2 = poly.vertices[(i + 1) % poly.vertices.size()];
      drawLineDDA(img, p1, p2, color, poly.thickness);
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////

static cv::Point lerp(const cv::Point &a, const cv::Point &b, double t) {
  return {int(a.x + (b.x - a.x) * t + .5), int(a.y + (b.y - a.y) * t + .5)};
}

cv::Point bezierPoint(const std::vector<cv::Point> &P, double t) {
  std::vector<cv::Point> tmp = P;
  for (int k = 1; k < P.size(); ++k)
    for (int i = 0; i + k < P.size(); ++i)
      tmp[i] = lerp(tmp[i], tmp[i + 1], t);
  return tmp[0];
}

void drawBezier(cv::Mat &img, const Bezier &bz) {
  const int STEPS = 100;
  std::vector<cv::Point> pts;
  for (int i = 0; i <= STEPS; ++i)
    pts.push_back(bezierPoint(bz.ctrl, i / (double)STEPS));
  for (int i = 0; i + 1 < pts.size(); ++i)
    drawLineDDA(img, pts[i], pts[i + 1], bz.color, bz.thickness);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void redraw_shapes(GtkWidget *widget) {
  image.setTo(cv::Scalar(255, 255, 255));
  for (const auto &line : lines)
    drawLineDDA(image, line.start, line.end, line.color, line.thickness);
  for (const auto &circle : circles)
    drawCircleMidpoint(image, circle.center, circle.radius, circle.color);
  for (const auto &poly : polygons)
    {
      float thickness = poly.thickness;
      if(clippingndex == -1)
        thickness = 1;
      drawPolygon(image, poly, poly.color);
    }

  for (auto &bz : beziers)
    drawBezier(image, bz);

  if (inBezierMode) {
    if (currentBezier.empty()) {
    } else if (currentBezier.size() == 1) {
      drawCircleMidpoint(image, currentBezier[0], 3, cv::Vec3b(200, 200, 200));
    } else {
      for (size_t i = 0; i + 1 < currentBezier.size(); ++i) {
        drawLineDDA(image, currentBezier[i], currentBezier[i + 1],
                    cv::Vec3b(200, 200, 200), 1);
      }
    }
  }
  draw_in_progress_polygon(image);
  gtk_widget_queue_draw(widget);
}

double distance_to_segment(cv::Point p, cv::Point a, cv::Point b) {
  cv::Point ab = b - a;
  double len_sq = ab.dot(ab);
  if (len_sq == 0)
    return cv::norm(p - a);
  double t = std::clamp((p - a).dot(ab) / len_sq, 0.0, 1.0);
  cv::Point proj = a + t * ab;
  return cv::norm(p - proj);
}

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

void try_complete_polygon(cv::Point pt, GtkWidget *widget) {
  const int closeThreshold = 10;
  if (!currentPolygonVertices.empty() &&
      cv::norm(pt - currentPolygonVertices[0]) < closeThreshold &&
      currentPolygonVertices.size() >= 3) {
    polygons.push_back({currentPolygonVertices, 1});
    currentPolygonVertices.clear();
    redraw_shapes(widget);
  } else {
    currentPolygonVertices.push_back(pt);
    redraw_shapes(widget);
  }
}

void try_delete_polygon(cv::Point pt, GtkWidget *widget) {
  const int threshold = 10;
  auto dist_to_segment = [](cv::Point p, cv::Point a, cv::Point b) -> double {
    cv::Point ab = b - a;
    double len_sq = ab.dot(ab);
    if (len_sq == 0)
      return cv::norm(p - a);
    double t = std::clamp((p - a).dot(ab) / len_sq, 0.0, 1.0);
    cv::Point proj = a + t * ab;
    return cv::norm(p - proj);
  };
  for (auto it = polygons.begin(); it != polygons.end(); ++it) {
    const auto &vertices = it->vertices;
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
    if (len_sq == 0)
      return cv::norm(p - a);
    double t = std::clamp((p - a).dot(ab) / len_sq, 0.0, 1.0);
    cv::Point proj = a + t * ab;
    return cv::norm(p - proj);
  };
  for (int i = 0; i < polygons.size(); ++i) {
    const auto &vertices = polygons[i].vertices;
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

bool is_point_near_vertex(const std::vector<cv::Point> &vertices, cv::Point pt,
                          int &outIndex, int threshold = 10) {
  for (int i = 0; i < vertices.size(); ++i) {
    if (cv::norm(pt - vertices[i]) < threshold) {
      outIndex = i;
      return true;
    }
  }
  return false;
}

bool is_point_near_edge_center(const std::vector<cv::Point> &vertices,
                               cv::Point pt, int &outEdgeIndex,
                               int threshold = 10) {
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

bool is_point_inside_polygon(const std::vector<cv::Point> &vertices,
                             cv::Point pt) {
  return cv::pointPolygonTest(vertices, pt, false) >= 0;
}

void move_polygon_vertex(Polygon &poly, int vertexIndex, cv::Point newPos) {
  if (vertexIndex >= 0 && vertexIndex < poly.vertices.size()) {
    poly.vertices[vertexIndex] = newPos;
  }
}

void move_polygon_edge(Polygon &poly, int edgeStartIndex,
                       cv::Point newMidpoint) {
  if (poly.vertices.size() < 2)
    return;
  cv::Point a = poly.vertices[edgeStartIndex];
  cv::Point b = poly.vertices[(edgeStartIndex + 1) % poly.vertices.size()];
  cv::Point currentMid = (a + b) / 2;
  cv::Point offset = newMidpoint - currentMid;
  poly.vertices[edgeStartIndex] += offset;
  poly.vertices[(edgeStartIndex + 1) % poly.vertices.size()] += offset;
}

void move_entire_polygon(Polygon &poly, cv::Point newPos) {
  cv::Point offset = newPos - storedClick;
  for (auto &v : poly.vertices)
    v += offset;
}

void handle_polygon_movement(cv::Point pt, GtkWidget *widget) {
  if (selectedPolygonIndex < 0 || selectedPolygonIndex >= polygons.size())
    return;

  Polygon &poly = polygons[selectedPolygonIndex];

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

  polygonMoveMode = PolygonMoveMode::None;
  selectedPolygonIndex = -1;
  selectedVertexIndex = -1;
  waitingForSecondClick = false;
  redraw_shapes(widget);
}

void handle_polygon_selection(cv::Point pt, GdkEventButton *event,
                              GtkWidget *widget) {
  const int threshold = 30;

  for (int i = 0; i < polygons.size(); ++i) {
    const auto &poly = polygons[i];

    if ((event->state & GDK_CONTROL_MASK) &&
        is_point_inside_polygon(poly.vertices, pt)) {
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

  try_complete_polygon(pt, widget);
}

void handle_polygon_click(cv::Point pt, GdkEventButton *event,
                          GtkWidget *widget) {
  if (waitingForSecondClick) {
    handle_polygon_movement(pt, widget);
  } else {
    handle_polygon_selection(pt, event, widget);
  }
}

gboolean draw_callback(GtkWidget *widget, cairo_t *cr, gpointer) {
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

bool try_select_circle(cv::Point pt, int threshold) {
  for (int i = 0; i < circles.size(); ++i) {
    if (std::abs(cv::norm(pt - circles[i].center) - circles[i].radius) <
        threshold) {
      selectedCircleIndex = i;
      selectedLineIndex = -1;
      return true;
    }
  }
  return false;
}

void try_delete_circle(cv::Point pt, GtkWidget *widget, int threshold) {
  auto it = std::find_if(circles.begin(), circles.end(), [&](const Circle &c) {
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

void try_delete_line(cv::Point pt, GtkWidget *widget, int threshold) {
  auto dist_to_segment = [](cv::Point p, cv::Point a, cv::Point b) -> double {
    cv::Point ab = b - a;
    double len_sq = ab.dot(ab);
    if (len_sq == 0)
      return cv::norm(p - a);
    double t = std::clamp((p - a).dot(ab) / len_sq, 0.0, 1.0);
    cv::Point proj = a + t * ab;
    return cv::norm(p - proj);
  };

  auto it = std::find_if(lines.begin(), lines.end(), [&](const Line &l) {
    return dist_to_segment(pt, l.start, l.end) < threshold;
  });
  if (it != lines.end()) {
    lines.erase(it);
    redraw_shapes(widget);
  }
}

void handle_circle_click(cv::Point pt, GdkEventButton *event, GtkWidget *widget,
                         int threshold) {
  static bool movingCircle = false;

  if (event->button == 1) {
    if (movingCircle && selectedCircleIndex >= 0 &&
        selectedCircleIndex < circles.size()) {
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

void handle_line_click(cv::Point pt, GdkEventButton *event, GtkWidget *widget,
                       int threshold) {
  static cv::Point lineStart;

  auto dist_to_segment = [](cv::Point p, cv::Point a, cv::Point b) -> double {
    cv::Point ab = b - a;
    double len_sq = ab.dot(ab);
    if (len_sq == 0)
      return cv::norm(p - a);
    double t = std::clamp((p - a).dot(ab) / len_sq, 0.0, 1.0);
    cv::Point proj = a + t * ab;
    return cv::norm(p - proj);
  };

  if (event->button == 1) {
    if (editMode != EditMode::None && selectedLineIndex >= 0 &&
        selectedLineIndex < lines.size()) {
      auto &line = lines[selectedLineIndex];
      if (editMode == EditMode::MoveStart)
        line.start = pt;
      else if (editMode == EditMode::MoveEnd)
        line.end = pt;
      editMode = EditMode::None;
      selectedLineIndex = -1;
      redraw_shapes(widget);
    } else if (try_select_line_endpoint(pt, threshold)) {
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
    auto it = std::find_if(lines.begin(), lines.end(), [&](const Line &l) {
      return dist_to_segment(pt, l.start, l.end) < threshold;
    });

    if (it != lines.end()) {
      selectedLineIndex = std::distance(lines.begin(), it);
      selectedCircleIndex = -1;
      std::cout << "Line selected with middle click for potential thickness "
                   "change.\n";
    }
  } else if (event->button == 3) {
    try_delete_line(pt, widget, threshold);
  }
}
bool isRectangleCW(const std::vector<cv::Point>& pts, double eps = 1e-6)
{
    if (pts.size() != 4)
        return false;

    const cv::Point2f &BL = pts[0],
                      &BR = pts[1],
                      &TR = pts[2],
                      &TL = pts[3];

    cv::Point2f v0 = BR - BL;  // bottom edge
    cv::Point2f v1 = TR - BR;  // right edge
    cv::Point2f v2 = TL - TR;  // top edge
    cv::Point2f v3 = BL - TL;  // left edge

    if (std::abs(v0.dot(v1)) > eps) return false;
    if (std::abs(v1.dot(v2)) > eps) return false;
    if (std::abs(v2.dot(v3)) > eps) return false;
    if (std::abs(v3.dot(v0)) > eps) return false;

    double Lbottom = cv::norm(v0);
    double Lright  = cv::norm(v1);
    double Ltop    = cv::norm(v2);
    double Lleft   = cv::norm(v3);

    if (std::abs(Lbottom - Ltop) > eps)  return false;
    if (std::abs(Lright  - Lleft) > eps) return false;

    if (Lbottom < eps || Lright < eps)
        return false;

    return true;
}

void handle_rectangle_click(cv::Point pt, GdkEventButton *event,
                            GtkWidget *widget) {
  const int thresh = 30;

  if (event->button == 1 && waitingForSecondClick &&
      polygonMoveMode == PolygonMoveMode::MoveWhole) {
    handle_polygon_movement(pt, widget);
    return;
  }

  if (event->button == 1 && (event->state & GDK_CONTROL_MASK)) {

    for (int i = 0; i < (int)polygons.size(); ++i) {
      if (polygons[i].vertices.size() == 4 &&
          is_point_inside_polygon(polygons[i].vertices, pt)) {

        polygonMoveMode = PolygonMoveMode::MoveWhole;
        selectedPolygonIndex = i;
        storedClick = pt;
        waitingForSecondClick = true;
        return;
      }
    }
  }

  if (event->button == 1 && awaitingRectCornerMove) {
    Polygon &rect = polygons[selectedRectIndex];
    int opp = (selectedRectVertexIndex + 2) % 4;
    cv::Point fixedPt = rect.vertices[opp];
    cv::Point newPt = pt;

    std::vector<cv::Point> v = {{fixedPt.x, fixedPt.y},
                                {fixedPt.x, newPt.y},
                                {newPt.x, newPt.y},
                                {newPt.x, fixedPt.y}};
    rect.vertices = v;
    awaitingRectCornerMove = false;
    selectedRectIndex = -1;
    selectedRectVertexIndex = -1;
    redraw_shapes(widget);
    return;
  }

  if (event->button == 1 && awaitingRectEdgeMove) {
    Polygon &rect = polygons[selectedRectIndex];
    int x0 = rect.vertices[0].x, y0 = rect.vertices[0].y,
        x1 = rect.vertices[2].x, y1 = rect.vertices[2].y,
        e = selectedRectEdgeIndex;
    switch (e) {
    case 0:
      x0 = pt.x;
      break; // left
    case 1:
      y1 = pt.y;
      break; // bottom
    case 2:
      x1 = pt.x;
      break; // right
    case 3:
      y0 = pt.y;
      break; // top
    }
    rect.vertices = {{x0, y0}, {x0, y1}, {x1, y1}, {x1, y0}};
    awaitingRectEdgeMove = false;
    selectedRectIndex = -1;
    selectedRectEdgeIndex = -1;
    redraw_shapes(widget);
    return;
  }

  if (event->button == 1) {
    for (int i = 0; i < (int)polygons.size(); ++i) {
      if (polygons[i].vertices.size() != 4)
        continue;
      int eidx;
      if (is_point_near_edge_center(polygons[i].vertices, pt, eidx, thresh)) {
        selectedRectIndex = i;
        selectedRectEdgeIndex = eidx;
        awaitingRectEdgeMove = true;
        return;
      }
    }
  }

  if (event->button == 1) {
    for (int i = 0; i < (int)polygons.size(); ++i) {
      if (polygons[i].vertices.size() != 4)
        continue;
      int vidx;
      if (is_point_near_vertex(polygons[i].vertices, pt, vidx, thresh)) {
        selectedRectIndex = i;
        selectedRectVertexIndex = vidx;
        awaitingRectCornerMove = true;
        return;
      }
    }
  }

  if (event->button == 1) {
    if (!awaitingRectSecond) {
      rectCorner = pt;
      awaitingRectSecond = true;
    } else {
      cv::Point a = rectCorner, b = pt;
      polygons.push_back(Polygon{
          {{a.x, a.y}, {a.x, b.y}, {b.x, b.y}, {b.x, a.y}}, 1, {255, 0, 0}});

          if (choose_clipping) {
            clippingndex     = polygons.size() - 1;
            choose_clipping  = false;
            std::cout << "Clipping rect is polygon #" 
                      << clippingndex << "\n";
        }
      awaitingRectSecond = false;
      redraw_shapes(widget);
    }
    return;
  }

  if (event->button == 2) {
    try_select_polygon(pt);
    return;
  }

  if (event->button == 3) {
    try_delete_polygon(pt, widget);
    return;
  }
}

gboolean on_mouse_click(GtkWidget *widget, GdkEventButton *event, gpointer) {
  cv::Point pt = map_click_to_image(widget, event->x, event->y);
  const int threshold = 15;
  if (inBezierMode) {
    if (event->button == 1) {
      currentBezier.push_back(pt);
      redraw_shapes(widget);
    }
    return TRUE;
  }

  if (choose_cliped
    && event->button == 2)    
{
    try_select_polygon(pt);  
    if (selectedPolygonIndex >= 0) {
        clipedIndex    = selectedPolygonIndex;
        choose_cliped  = false;
        std::cout << "Will clip polygon #"
                  << clipedIndex
                  << " against rect #"
                  << clippingndex << "\n";
    }
    return TRUE;  
}

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
  case ShapeType::Rectangle:
    handle_rectangle_click(pt, event, widget);
    break;
  }

  return TRUE;
}

bool adjust_line_thickness(double delta) {
  if (selectedLineIndex >= 0 && selectedLineIndex < lines.size()) {
    auto &line = lines[selectedLineIndex];
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
    auto &circle = circles[selectedCircleIndex];
    if (delta < 0)
      circle.radius = std::min(300, circle.radius + 5);
    else
      circle.radius = std::max(5, circle.radius - 5);
    std::cout << "Circle radius changed to: " << circle.radius << "\n";
    return true;
  }
  return false;
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

static void on_clip_activate(GtkMenuItem *item, gpointer user_data)
{
  if (clippingndex < 0 || clipedIndex < 0) {
    std::cerr << "Need both a clipping rect and a polygon selected!\n";
    return;
  }

  if (!isRectangleCW(polygons[clippingndex].vertices) || polygons[clipedIndex].thickness != 1) {
    std::cout << "The polygons must be elidgible for clipping!\n";
    return;
  }

  const auto &rectPoly = polygons[clippingndex];
  const auto &srcPoly  = polygons[clipedIndex];
  Polygon resultPolyInside;    

  int xmin = rectPoly.vertices[0].x, xmax = xmin;
  int ymin = rectPoly.vertices[0].y, ymax = ymin;
  for (auto &v : rectPoly.vertices) {
    xmin = std::min(xmin, v.x);
    xmax = std::max(xmax, v.x);
    ymin = std::min(ymin, v.y);
    ymax = std::max(ymax, v.y);
  }

  // colours
  cv::Vec3b clrOutside  = srcPoly.color; 
  cv::Vec3b white = {255, 255, 255};
  cv::Vec3b clrInside = white - clrOutside; 

  for (size_t i = 0; i < srcPoly.vertices.size(); ++i) {
    cv::Point P0 = srcPoly.vertices[i];
    cv::Point P1 = srcPoly.vertices[(i+1) % srcPoly.vertices.size()];

    cv::Point C0, C1;
    bool accept = cohenSutherlandClip(P0, P1,
                                      xmin, ymin, xmax, ymax,
                                      C0, C1);

    if (accept) {
      
      drawLineDDA(image, C0, C1, clrInside, 2*srcPoly.thickness);

      if ((C0 != P0)){
        
        drawLineDDA(image, P0, C0, clrOutside, 2*srcPoly.thickness);
      }

      if ((C1 != P1)){
        
        drawLineDDA(image, C1, P1, clrOutside, 2*srcPoly.thickness);
      }
    } else {

      drawLineDDA(image, P0, P1, clrOutside, 2*srcPoly.thickness);
    }
  }

  gtk_widget_queue_draw(drawing_area);
}

gboolean on_scroll(GtkWidget *widget, GdkEventScroll *event, gpointer) {
  double delta = extract_scroll_delta(event);
  if (delta == 0)
    return TRUE;

  bool changed = false;

  // Polygon thickness
  if (selectedPolygonIndex >= 0 && selectedPolygonIndex < polygons.size()) {
    auto &poly = polygons[selectedPolygonIndex];
    if (delta < 0)
      poly.thickness = std::min(50, poly.thickness + 1);
    else
      poly.thickness = std::max(1, poly.thickness - 1);
    std::cout << "Polygon thickness changed to: " << poly.thickness << "\n";
    changed = true;
  }

  // Circle radius
  else if (selectedCircleIndex >= 0 && selectedCircleIndex < circles.size()) {
    auto &circle = circles[selectedCircleIndex];
    if (delta < 0)
      circle.radius = std::min(300, circle.radius + 5);
    else
      circle.radius = std::max(5, circle.radius - 5);
    std::cout << "Circle radius changed to: " << circle.radius << "\n";
    changed = true;
  }

  // Line thickness
  else if (selectedLineIndex >= 0 && selectedLineIndex < lines.size()) {
    auto &line = lines[selectedLineIndex];
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

void on_shape_selected(GtkComboBoxText *combo, gpointer) {
  const gchar *selected = gtk_combo_box_text_get_active_text(combo);
  if (g_strcmp0(selected, "Line") == 0)
    currentShape = ShapeType::Line;
  else if (g_strcmp0(selected, "Circle") == 0)
    currentShape = ShapeType::Circle;
  std::cout << "Shape changed to: " << selected << std::endl;
}

void clear_all_shapes(GtkWidget *widget) {
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
  beziers.clear();
  currentBezier.clear();
  inBezierMode = false;
  std::cout << "Canvas cleared.\n";
  redraw_shapes(widget);
}

cv::Vec3b open_color_chooser(GtkWidget *parent) {
  GtkWidget *dialog =
      gtk_color_chooser_dialog_new("Choose Color", GTK_WINDOW(parent));
  cv::Vec3b result{255, 255, 255}; 

  if (gtk_dialog_run(GTK_DIALOG(dialog)) == GTK_RESPONSE_OK) {
    GdkRGBA gdk_color;
    gtk_color_chooser_get_rgba(GTK_COLOR_CHOOSER(dialog), &gdk_color);
    result = {static_cast<uchar>(gdk_color.blue * 255),
              static_cast<uchar>(gdk_color.green * 255),
              static_cast<uchar>(gdk_color.red * 255)};
  }

  gtk_widget_destroy(dialog);
  return result;
}

void change_selected_shape_color(GtkWidget *parent) {
  cv::Vec3b newColor = open_color_chooser(parent);

  if (selectedLineIndex >= 0 && selectedLineIndex < lines.size()) {
    lines[selectedLineIndex].color = newColor;
  } else if (selectedCircleIndex >= 0 && selectedCircleIndex < circles.size()) {
    circles[selectedCircleIndex].color = newColor;
  } else if (selectedPolygonIndex >= 0 &&
             selectedPolygonIndex < polygons.size()) {
    polygons[selectedPolygonIndex].color = newColor;
  }

  redraw_shapes(parent);
}

void save_shapes_to_file(GtkWidget *parent) {
  GtkWidget *dialog = gtk_file_chooser_dialog_new(
      "Save Shapes", GTK_WINDOW(parent), GTK_FILE_CHOOSER_ACTION_SAVE,
      "_Cancel", GTK_RESPONSE_CANCEL, "_Save", GTK_RESPONSE_ACCEPT, NULL);

  gtk_file_chooser_set_do_overwrite_confirmation(GTK_FILE_CHOOSER(dialog),
                                                 TRUE);
  gtk_file_chooser_set_current_name(GTK_FILE_CHOOSER(dialog), "shapes.vec");

  if (gtk_dialog_run(GTK_DIALOG(dialog)) == GTK_RESPONSE_ACCEPT) {
    char *filename = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dialog));
    std::ofstream out(filename);
    if (!out) {
      std::cerr << "Failed to open file for saving.\n";
      g_free(filename);
      gtk_widget_destroy(dialog);
      return;
    }

    for (const auto &line : lines)
      out << "LINE " << line.start.x << " " << line.start.y << " " << line.end.x
          << " " << line.end.y << " " << line.thickness << " "
          << (int)line.color[0] << " " << (int)line.color[1] << " "
          << (int)line.color[2] << "\n";

    for (const auto &circle : circles)
      out << "CIRCLE " << circle.center.x << " " << circle.center.y << " "
          << circle.radius << " " << (int)circle.color[0] << " "
          << (int)circle.color[1] << " " << (int)circle.color[2] << "\n";

    for (const auto &poly : polygons) {
      out << "POLYGON "
          << poly.thickness << " "
          << poly.vertices.size();
      for (const auto &v : poly.vertices)
        out << " " << v.x << " " << v.y;
      out << " "
          << int(poly.color[0]) << " "
          << int(poly.color[1]) << " "
          << int(poly.color[2]) << " "
          << int(poly.filled) << " "
          << int(poly.useImageFill)
          << "\n";

      if (poly.useImageFill && !poly.fillImage.empty()) {
        int rows = poly.fillImage.rows;
        int cols = poly.fillImage.cols;
        out << "IMG_FILL " << rows << " " << cols << "\n";
        for (int y = 0; y < rows; ++y) {
          for (int x = 0; x < cols; ++x) {
            cv::Vec3b pix = poly.fillImage.at<cv::Vec3b>(y, x);
            out << int(pix[0]) << " "
                << int(pix[1]) << " "
                << int(pix[2]) << "\n";
          }
        }
      }
    }
    

    for (const auto &bz : beziers) {
      out << "BEZIER " << bz.thickness << " " << bz.ctrl.size();
      for (const auto &p : bz.ctrl)
        out << " " << p.x << " " << p.y;
      out << " " << (int)bz.color[0] << " " << (int)bz.color[1] << " "
          << (int)bz.color[2] << "\n";
    }

    out.close();
    g_free(filename);
  }

  gtk_widget_destroy(dialog);
}

void load_shapes_from_file(GtkWidget *parent) {
  GtkWidget *dialog = gtk_file_chooser_dialog_new(
      "Load Shapes", GTK_WINDOW(parent), GTK_FILE_CHOOSER_ACTION_OPEN,
      "_Cancel", GTK_RESPONSE_CANCEL, "_Open", GTK_RESPONSE_ACCEPT, NULL);

  if (gtk_dialog_run(GTK_DIALOG(dialog)) == GTK_RESPONSE_ACCEPT) {
    char *filename = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dialog));
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
        int r, g, b;
        in >> l.start.x >> l.start.y >> l.end.x >> l.end.y >> l.thickness >>
            r >> g >> b;
        l.color = cv::Vec3b(r, g, b); 
        lines.push_back(l);
      } else if (type == "CIRCLE") {
        Circle c;
        int r, g, b;
        bool filled;
        in >> c.center.x >> c.center.y >> c.radius >> r >> g >> b;
        c.color = cv::Vec3b(r, g, b);
        circles.push_back(c);
      } else if (type == "POLYGON") {
        Polygon p;
        int count, r, g, b, imgFlag;
        in >> p.thickness >> count;
        for (int i = 0; i < count; ++i) {
          cv::Point pt;
          in >> pt.x >> pt.y;
          p.vertices.push_back(pt);
        }
        in >> r >> g >> b;
        p.color = cv::Vec3b(r, g, b);
        in >> p.filled;
        in >> imgFlag;
        p.useImageFill = (imgFlag != 0);

        if (p.useImageFill) {
          std::string marker;
          int rows, cols;
          in >> marker;
          if (marker != "IMG_FILL") {
            throw std::runtime_error("Expected IMG_FILL marker, got: " + marker);
          }
          in >> rows >> cols;
          p.fillImage = cv::Mat(rows, cols, CV_8UC3);
          for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
              int bb, gg, rr;
              in >> rr >> gg >> bb;
              p.fillImage.at<cv::Vec3b>(y, x) =
                cv::Vec3b((uint8_t)rr, (uint8_t)gg, (uint8_t)bb);
            }
          }
        }

        polygons.push_back(std::move(p));
      
      } else if (type == "BEZIER") {
        Bezier bz;
        int count, r, g, b;
        in >> bz.thickness >> count;
        bz.ctrl.clear();
        for (int i = 0; i < count; ++i) {
          cv::Point pt;
          in >> pt.x >> pt.y;
          bz.ctrl.push_back(pt);
        }
        in >> r >> g >> b;

        bz.color = cv::Vec3b(r, g, b);
        beziers.push_back(bz);
      }
    }

    in.close();
    redraw_shapes(parent);
    g_free(filename);
  }

  gtk_widget_destroy(dialog);
}

GtkWidget *create_shape_menu(GtkWidget *window) {
  GtkWidget *menu_bar = gtk_menu_bar_new();

  // === FILE MENU ===
  GtkWidget *file_menu_root = gtk_menu_item_new_with_label("File");
  GtkWidget *file_submenu = gtk_menu_new();

  GtkWidget *clear_item = gtk_menu_item_new_with_label("Clear");
  GtkWidget *save_item = gtk_menu_item_new_with_label("Save");
  GtkWidget *load_item = gtk_menu_item_new_with_label("Load");
  GtkWidget *color_item = gtk_menu_item_new_with_label("Change Color");

  gtk_menu_shell_append(GTK_MENU_SHELL(file_submenu), clear_item);
  gtk_menu_shell_append(GTK_MENU_SHELL(file_submenu), save_item);
  gtk_menu_shell_append(GTK_MENU_SHELL(file_submenu), load_item);
  gtk_menu_shell_append(GTK_MENU_SHELL(file_submenu), color_item);

  gtk_menu_item_set_submenu(GTK_MENU_ITEM(file_menu_root), file_submenu);
  gtk_menu_shell_append(GTK_MENU_SHELL(menu_bar), file_menu_root);

  // === SHAPES MENU ===
  GtkWidget *shape_menu_root = gtk_menu_item_new_with_label("Shapes");
  GtkWidget *shape_submenu = gtk_menu_new();

  GtkWidget *line_item = gtk_menu_item_new_with_label("Line");
  GtkWidget *circle_item = gtk_menu_item_new_with_label("Circle");
  GtkWidget *polygon_item = gtk_menu_item_new_with_label("Polygon");
  GtkWidget *aa_toggle_item =
      gtk_check_menu_item_new_with_label("Enable Anti-Aliasing");
  GtkWidget *bezier_toggle =
      gtk_check_menu_item_new_with_label("Bezier Curve Mode");
  GtkWidget *rect_item = gtk_menu_item_new_with_label("Rectangle");

  gtk_menu_shell_append(GTK_MENU_SHELL(shape_submenu), line_item);
  gtk_menu_shell_append(GTK_MENU_SHELL(shape_submenu), circle_item);
  gtk_menu_shell_append(GTK_MENU_SHELL(shape_submenu), polygon_item);
  gtk_menu_shell_append(GTK_MENU_SHELL(shape_submenu), aa_toggle_item);
  gtk_menu_shell_append(GTK_MENU_SHELL(shape_submenu), bezier_toggle);
  gtk_menu_shell_append(GTK_MENU_SHELL(shape_submenu), rect_item);

  gtk_menu_item_set_submenu(GTK_MENU_ITEM(shape_menu_root), shape_submenu);
  gtk_menu_shell_append(GTK_MENU_SHELL(menu_bar), shape_menu_root);

  // === CLIPING MENU ===
  GtkWidget *clip_menu_root = gtk_menu_item_new_with_label("Clipping");
  GtkWidget *clip_submenu = gtk_menu_new();

  GtkWidget *clipped_item = gtk_menu_item_new_with_label("Clipped");
  GtkWidget *clipping_item = gtk_menu_item_new_with_label("Clipping");
  GtkWidget *clip_item = gtk_menu_item_new_with_label("Clip");

  gtk_menu_shell_append(GTK_MENU_SHELL(clip_submenu), clipped_item);
  gtk_menu_shell_append(GTK_MENU_SHELL(clip_submenu), clipping_item);
  gtk_menu_shell_append(GTK_MENU_SHELL(clip_submenu), clip_item);

  gtk_menu_item_set_submenu(GTK_MENU_ITEM(clip_menu_root), clip_submenu);
  gtk_menu_shell_append(GTK_MENU_SHELL(menu_bar), clip_menu_root);

  // === FILLING MENU ===
  GtkWidget *fill_menu_root = gtk_menu_item_new_with_label("Filling");
  GtkWidget *fill_submenu = gtk_menu_new();

  GtkWidget *fill_item = gtk_menu_item_new_with_label("Fill");
  GtkWidget *unfill_item = gtk_menu_item_new_with_label("Unfill");
  GtkWidget *loadImgFill_item = gtk_menu_item_new_with_label("Load Image Fill");

  gtk_menu_shell_append(GTK_MENU_SHELL(fill_submenu), fill_item);
  gtk_menu_shell_append(GTK_MENU_SHELL(fill_submenu), unfill_item);
  gtk_menu_shell_append(GTK_MENU_SHELL(fill_submenu), loadImgFill_item);

  gtk_menu_item_set_submenu(GTK_MENU_ITEM(fill_menu_root), fill_submenu);
  gtk_menu_shell_append(GTK_MENU_SHELL(menu_bar), fill_menu_root);

  // === SIGNAL CONNECTIONS ===
  
    g_signal_connect(line_item, "activate",
                     G_CALLBACK(+[](GtkWidget *, gpointer) {
                       currentShape = ShapeType::Line;
                       selectedCircleIndex = -1;
                       selectedPolygonIndex = -1;
                       std::cout << "Shape changed to: Line" << std::endl;
                     }),
                     NULL);

    g_signal_connect(circle_item, "activate",
                     G_CALLBACK(+[](GtkWidget *, gpointer) {
                       currentShape = ShapeType::Circle;
                       selectedLineIndex = -1;
                       selectedPolygonIndex = -1;
                       std::cout << "Shape changed to: Circle" << std::endl;
                     }),
                     NULL);

    g_signal_connect(polygon_item, "activate",
                     G_CALLBACK(+[](GtkWidget *, gpointer) {
                       currentShape = ShapeType::Polygon;
                       selectedLineIndex = -1;
                       selectedCircleIndex = -1;
                       std::cout << "Shape changed to: Polygon" << std::endl;
                     }),
                     NULL);

    g_signal_connect(aa_toggle_item, "toggled",
                     G_CALLBACK(+[](GtkCheckMenuItem *item, gpointer) {
                       use_antialiasing = gtk_check_menu_item_get_active(item);
                       std::cout << "Anti-aliasing: "
                                 << (use_antialiasing ? "ON" : "OFF")
                                 << std::endl;
                       redraw_shapes(drawing_area);
                     }),
                     NULL);

    g_signal_connect(clear_item, "activate",
                     G_CALLBACK(+[](GtkWidget *, gpointer) {
                       clear_all_shapes(drawing_area);
                     }),
                     NULL);

    g_signal_connect(save_item, "activate",
                     G_CALLBACK(+[](GtkWidget *, gpointer user_data) {
                       GtkWidget *win = GTK_WIDGET(user_data);
                       save_shapes_to_file(win);
                     }),
                     window);

    g_signal_connect(load_item, "activate",
                     G_CALLBACK(+[](GtkWidget *, gpointer user_data) {
                       GtkWidget *win = GTK_WIDGET(user_data);
                       load_shapes_from_file(win);
                     }),
                     window);

    g_signal_connect(color_item, "activate",
                     G_CALLBACK(+[](GtkWidget *, gpointer user_data) {
                       GtkWidget *win = GTK_WIDGET(user_data);
                       change_selected_shape_color(win);
                     }),
                     window);
    g_signal_connect(bezier_toggle, "toggled",
                     G_CALLBACK(+[](GtkCheckMenuItem *m, gpointer) {
                       inBezierMode = gtk_check_menu_item_get_active(m);
                       if (!inBezierMode && currentBezier.size() >= 2) {
                         beziers.push_back({currentBezier, 1, {0, 0, 0}});
                         currentBezier.clear();
                         redraw_shapes(drawing_area);
                       }
                     }),
                     nullptr);

    g_signal_connect(
        rect_item, "activate", G_CALLBACK(+[](GtkWidget *, gpointer) {
          currentShape = ShapeType::Rectangle;
          selectedLineIndex = selectedCircleIndex = selectedPolygonIndex = -1;
          std::cout << "Shape changed to: Rectangle\n";
        }),
        nullptr);

    g_signal_connect(clip_item, "activate",
                      G_CALLBACK(on_clip_activate),
                     nullptr);
    g_signal_connect(clipping_item, "activate",
                      G_CALLBACK(+[](GtkWidget*, gpointer){
                        choose_clipping = true;
                        choose_cliped   = false;
                        clippingndex    = -1;
                        clipedIndex     = -1;
                        std::cout << "Now draw a rectangle for clipping window.\n";
                      }),
                      nullptr);
                  
    g_signal_connect(clipped_item, "activate",
        G_CALLBACK(+[](GtkWidget*, gpointer){
          choose_cliped   = true;
          choose_clipping = false;
          clipedIndex     = -1;
          std::cout << "Now middle-click the polygon you wish to clip.\n";
        }),
        nullptr);
    g_signal_connect(fill_item, "activate",
        G_CALLBACK(+[](GtkWidget*, gpointer){
          if (selectedPolygonIndex >= 0 && selectedPolygonIndex < polygons.size()) {
            polygons[selectedPolygonIndex].filled = true;
            redraw_shapes(drawing_area);
          }
        }),
        nullptr);
    g_signal_connect(unfill_item, "activate",
        G_CALLBACK(+[](GtkWidget*, gpointer){
          if (selectedPolygonIndex >= 0 && selectedPolygonIndex < polygons.size()) {
            polygons[selectedPolygonIndex].filled = false;
            redraw_shapes(drawing_area);
          }
        }),
        nullptr);
    g_signal_connect(loadImgFill_item, "activate",
          G_CALLBACK(+[](GtkWidget *w, gpointer) {
            if (selectedPolygonIndex<0) return;
            GtkWidget* dlg = gtk_file_chooser_dialog_new(
                "Pick Fill Image", GTK_WINDOW(w),
                GTK_FILE_CHOOSER_ACTION_OPEN,
                "_Cancel", GTK_RESPONSE_CANCEL,
                "_Open",   GTK_RESPONSE_ACCEPT,
                NULL);
            if (gtk_dialog_run(GTK_DIALOG(dlg))==GTK_RESPONSE_ACCEPT) {
              char *fname = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dlg));
              polygons[selectedPolygonIndex].fillImage = cv::imread(fname, cv::IMREAD_COLOR);
              polygons[selectedPolygonIndex].useImageFill = true;
              g_free(fname);
              redraw_shapes(drawing_area);
            }
            gtk_widget_destroy(dlg);
          }), nullptr);
  
  return menu_bar;
}

int main(int argc, char *argv[]) {
  gtk_init(&argc, &argv);
  image = cv::Mat::zeros(600, 800, CV_8UC3);
  image.setTo(cv::Scalar(255, 255, 255));
  GtkWidget *window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
  gtk_window_set_title(GTK_WINDOW(window), "Line Editor with Scaled Thickness");
  gtk_window_set_default_size(GTK_WINDOW(window), 1000, 800);

  GtkWidget *vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
  gtk_container_add(GTK_CONTAINER(window), vbox);

  shape_menu = create_shape_menu(window);
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

  gtk_widget_show_all(window);
  gtk_main();
  return 0;
}
