//
// Created by anonymous author on 2022/9/27.
//

#include "triangle.h"


void Triangle::EnumTriangle(const DataGraph &din, const DataGraph &dout) {
    ui n = dout.getNumVertices(), m = dout.getNumEdges();
    const EdgeID *inOffset = din.getOffsets();
    const VertexID *inNbors = din.getNbors();
    const EdgeID *outOffset = dout.getOffsets();
    const VertexID *outNbors = dout.getNbors();
    vertex1Cnt = new Count[n];
    vertex2Cnt = new Count[n];
    vertex3Cnt = new Count[n];
    memset(vertex1Cnt, 0, sizeof(Count) * n);
    memset(vertex2Cnt, 0, sizeof(Count) * n);
    memset(vertex3Cnt, 0, sizeof(Count) * n);
    edge1Cnt = new Count[m];
    edge2Cnt = new Count[m];
    edge3Cnt = new Count[m];
    memset(edge1Cnt, 0, sizeof(Count) * m);
    memset(edge2Cnt, 0, sizeof(Count) * m);
    memset(edge3Cnt, 0, sizeof(Count) * m);
    // get triangle counts and allocate memory
    for (VertexID v1 = 0; v1 < dout.getNumVertices(); ++v1) {
        for (EdgeID e12 = outOffset[v1]; e12 < outOffset[v1 + 1]; ++e12) {
            // v1 -> v2
            VertexID v2 = outNbors[e12];
            for (EdgeID e23 = outOffset[v2]; e23 < outOffset[v2 + 1]; ++e23) {
                // v2 -> v3
                VertexID v3 = outNbors[e23];
                EdgeID e13 = dout.getEdgeID(v1, v3);
                if (e13 > m) continue;
                // v1 -> v3. must not be v3 -> v1 due to acyclic
                ++vertex1Cnt[v2];
                ++vertex2Cnt[v3];
                ++vertex3Cnt[v1];
                ++edge1Cnt[e23];
                ++edge2Cnt[e13];
                ++edge3Cnt[e12];
            }
        }
    }
    offset1 = new EdgeID[m + 1];
    offset2 = new EdgeID[m + 1];
    offset3 = new EdgeID[m + 1];
    offset1[0] = 0;
    triangleCnt = 0;
    for (ui i = 0; i < m; ++i) {
        triangleCnt += edge1Cnt[i];
        triangleCnt += edge2Cnt[i];
        triangleCnt += edge3Cnt[i];
    }
    for (ui i = 0; i < m; ++i) {
        offset2[i] = offset1[i] + edge1Cnt[i];
        offset3[i] = offset2[i] + edge2Cnt[i];
        offset1[i + 1] = offset3[i] + edge3Cnt[i];
    }
    offset1[m] = triangleCnt;
    offset2[m] = triangleCnt;
    offset3[m] = triangleCnt;
    triend = new VertexID[triangleCnt];
    // store triangles
    for (VertexID v1 = 0; v1 < dout.getNumVertices(); ++v1) {
        for (EdgeID e12 = outOffset[v1]; e12 < outOffset[v1 + 1]; ++e12) {
            // v1 -> v2
            VertexID v2 = outNbors[e12];
            VertexID pos = offset1[e12];
            for (EdgeID e23 = inOffset[v2]; e23 < inOffset[v2 + 1]; ++e23) {
                // v3 -> v2
                VertexID v3 = inNbors[e23];
                if (v1 == v3) continue;
                int e13 = dout.getEdgeID(v1, v3);
                if (e13 != -1) {
                    triend[pos] = v3;
                    ++pos;
                }
            }
            for (EdgeID e23 = outOffset[v2]; e23 < outOffset[v2 + 1]; ++e23) {
                // v2 -> v3
                VertexID v3 = outNbors[e23];
                int e13 = dout.getEdgeID(v1, v3);
                if (e13 != -1) {
                    // v1 -> v3
                    triend[pos] = v3;
                    ++pos;
                }
            }
        }
    }
}

VertexID *Triangle::getTriend(EdgeID &e, int type, ui &cnt) const {
    if (type == 1) {
        cnt = offset2[e] - offset1[e];
        return triend + offset1[e];
    }
    else if (type == 2) {
        cnt = offset3[e] - offset2[e];
        return triend + offset2[e];
    }
    else if (type == 3) {
        cnt = offset1[e + 1] - offset3[e];
        return triend + offset3[e];
    }
    else {
        cnt = offset1[e + 1] - offset1[e];
        return triend + offset1[e];
    }
}

double Triangle::memoryCost() const {
    double mem = (double)(triangleCnt) * sizeof(VertexID);
    return mem / 1000000000;
}

// m is the number of directed edges
void Triangle::load(const std::string &path, ui m) {
    std::ifstream in(path, std::ios::in | std::ios::binary);
    if (!in.is_open()) {
        std::cout << "Can not open file " << path << "!" << std::endl;
        exit(1);
    }
    offset1 = new EdgeID[m + 1];
    offset2 = new EdgeID[m + 1];
    offset3 = new EdgeID[m + 1];
    in.read(reinterpret_cast<char*>(offset1), sizeof(EdgeID) * (m + 1));
    in.read(reinterpret_cast<char*>(offset2), sizeof(EdgeID) * (m + 1));
    in.read(reinterpret_cast<char*>(offset3), sizeof(EdgeID) * (m + 1));
    in.read(reinterpret_cast<char*>(&triangleCnt), sizeof(Count));
    triend = new VertexID[triangleCnt];
    in.read(reinterpret_cast<char*>(triend), sizeof(VertexID) * triangleCnt);
    in.close();
}

void Triangle::save(const std::string &path, ui m) const {
    std::ofstream out(path, std::ios::out | std::ios::binary);
    // write offsets
    out.write(reinterpret_cast<const char*>(offset1), sizeof(EdgeID) * (m + 1));
    out.write(reinterpret_cast<const char*>(offset2), sizeof(EdgeID) * (m + 1));
    out.write(reinterpret_cast<const char*>(offset3), sizeof(EdgeID) * (m + 1));
    // write triends
    out.write(reinterpret_cast<const char*>(&triangleCnt), sizeof(Count));
    out.write(reinterpret_cast<const char*>(triend), sizeof(VertexID) * triangleCnt);
    out.close();
}
