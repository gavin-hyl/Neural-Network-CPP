#include "Read.h"

using std::vector;
using std::string;

csvFile::csvFile(string path, string flags) {
    file.open(path);
    string line;
    string token;
    std::stringstream streamLine;

    if (file.is_open()) {
        std::getline(file, line);
        streamLine = std::stringstream(line);
        while (std::getline(streamLine, token, ',')) {
            headers.push_back(token);
        }
        while (std::getline(file, line)) {
            vector<string> lineVector;
            streamLine = std::stringstream(line);
            while (std::getline(streamLine, token, ',')) {
                lineVector.push_back(token);
            }
            data.push_back(lineVector);
        }
    } else {
        std::cerr << "[ERROR] (csvFile): File could not be opened." << std::endl;
        exit(EXIT_FAILURE);
    }
}

vector<string> csvFile::getStringRow(int row) {
    return data.at(row);
}

vector<double> csvFile::getDoubleRow(int row) {
    vector<double> doubleRow;
    vector<string> strRow = data.at(row);
    int size = strRow.size();
    for (int i = 0; i < size; i++) {
        doubleRow.push_back(std::stod(strRow.at(i)));
    }
    return doubleRow;
}

array<int, 2> csvFile::getDimensions() {
    return {int(data.size()), int(data[0].size())};
}


vector<string> csvFile::getColumnHeaders() {
    return headers;
}