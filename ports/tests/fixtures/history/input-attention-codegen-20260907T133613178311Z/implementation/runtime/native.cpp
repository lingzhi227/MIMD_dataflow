#include "spatial.hpp"
#ifndef MW_MAX_INPUT
#define MW_MAX_INPUT 262144
#endif
void design();
int main() {
  try {
    std::cout << std::setprecision(9);
    int epochs = 0;
    if (!(std::cin >> epochs) || epochs != MW_EPOCHS)
      throw std::runtime_error("epoch contract");
    for (int e = 0; e < epochs; ++e) {
      int ports = 0;
      if (!(std::cin >> ports) || ports < 1 || ports > 48)
        throw std::runtime_error("port count");
      spatial::inputs.clear();
      spatial::index_inputs.clear();
      spatial::outputs.clear();
      spatial::index_outputs.clear();
      for (int p = 0; p < ports; ++p) {
        std::string name;
        int n = 0;
        if (!(std::cin >> name)) throw std::runtime_error("input name");
        bool index = name == "@u32";
        if (index && !(std::cin >> name)) throw std::runtime_error("index name");
        if (!(std::cin >> n) || n < 1 || n > MW_MAX_INPUT)
          throw std::runtime_error("input extent");
        if (index) {
          std::vector<uint32_t> indices(n);
          for (auto &x : indices) {
            std::string token;
            if (!(std::cin >> token) || token.empty() || token.find_first_not_of("0123456789") != std::string::npos)
              throw std::runtime_error("unsigned integer input");
            auto wide=std::stoull(token);
            if (wide>UINT32_MAX) throw std::runtime_error("u32 input overflow");
            x=static_cast<uint32_t>(wide);
          }
          if (spatial::inputs.count(name) || !spatial::index_inputs.emplace(name, indices).second)
            throw std::runtime_error("duplicate input");
          continue;
        }
        std::vector<float> v(n);
        for (auto &x : v) {
          // Some libc++ float extractors reject representable binary32 subnormals.
          // Parse at wider precision, validate, then perform the explicit f32 conversion.
          double value;
          if (!(std::cin >> value) || !std::isfinite(value) || value < -MW_BOUND || value > MW_BOUND)
            throw std::runtime_error("input value contract");
          x=static_cast<float>(value);
        }
        if (spatial::index_inputs.count(name) || !spatial::inputs.emplace(name, v).second)
          throw std::runtime_error("duplicate input");
      }
      if (!std::cin)
        throw std::runtime_error("input format");
      design();
      std::cout << "epoch " << e << "\n";
      for (auto &entry : spatial::index_outputs) {
        std::cout << "@u32 " << entry.first << " " << entry.second.size();
        for (auto x : entry.second) std::cout << " " << x;
        std::cout << "\n";
      }
      for (auto &entry : spatial::outputs) {
        std::cout << entry.first << " " << entry.second.size();
        for (auto x : entry.second)
          std::cout << " " << x;
        std::cout << "\n";
      }
    }
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
