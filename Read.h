#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <array>

#define MAX_TOKEN 64
#define MAX_LINE 16384

using std::string;
using std::vector;
using std::array;

class csvFile
{
private:
    std::fstream file;
    vector<string> headers;
    vector<vector<string>> data;

public:
    csvFile(string path, string flags);
    vector<string> getColumnHeaders();
    vector<string> getStringRow(int row);
    vector<double> getDoubleRow(int row);
    array<int, 2> getDimensions();
};