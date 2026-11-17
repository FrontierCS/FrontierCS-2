
#include <string>

class Pattern {
public:
  Pattern() {}
  Pattern(std::string name) : name_(name) { }
  ~Pattern() {}
  bool is_diamond() { return name_ == "diamond"; }
  bool is_9clique() { return name_ == "9clique"; }
  bool is_6clique() { return name_ == "6clique"; }
  bool is_4clique() { return name_ == "4clique"; }
  bool is_5clique() { return name_ == "5clique"; }
  bool is_triangle() { return name_ == "triangle"; }
  bool is_rectangle() { return name_ == "rectangle"; }
  bool is_pentagon() { return name_ == "pentagon"; }
  bool is_house() { return name_ == "house"; }
  bool is_triangletriangle() { return name_ == "triangletriangle" || name_ == "dumbell"; }
  bool is_5path() { return name_ == "5path"; }
  std::string to_string() { return name_; }
  std::string get_name() { return name_; }
  int get_nodes() { 
    if(is_diamond() || is_rectangle() || is_4clique()) { 
      return 4;
    } else if(is_5clique() || is_house() || is_pentagon() || is_5path()) { 
      return 5; 
    } else if(is_triangle()) {
      return 3;
    } else if (is_6clique() || is_triangletriangle()) {
      return 6;
    } else {
      return 9;
    }
  }
private:
  std::string name_;
};

