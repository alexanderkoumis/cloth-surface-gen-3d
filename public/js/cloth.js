  'use strict'

class Cloth {

  constructor(blockDim, boundingBox, moveScalar, pointSize, minFillRatio) {

    function calcGridDim(boundingBox, blockDim) {
      let bBoxSize = boundingBox.max.clone().sub(boundingBox.min);
      return new THREE.Vector3(
        parseInt(bBoxSize.x / blockDim.x),
        1,
        parseInt(bBoxSize.z / blockDim.z)
      );
    }

    this.blockDim = blockDim;
    this.gridDim = calcGridDim(boundingBox, blockDim);
    this.boundingBox = boundingBox;
    this.moveScalar = moveScalar;
    this.pointSize = pointSize;
    this.vertices = [];

    let geometryPoints = new THREE.Geometry();
    for (let z = 0; z < this.gridDim.z; ++z) {
      for (let x = 0; x < this.gridDim.x; ++x) {
        let position = new THREE.Vector3(x * blockDim.x + boundingBox.min.x,
                                        boundingBox.max.y,
                                        z * blockDim.z + boundingBox.min.z);
        let moveVec = new THREE.Vector3(0, -moveScalar, 0);
        let fromSplit = false;
        geometryPoints.vertices.push(position);
        this.vertices.push(new Vertice(position, moveVec, fromSplit,
                                       minFillRatio));
      }
    }

    let geometryMesh = new THREE.Geometry();
    geometryMesh.vertices = geometryPoints.vertices;

    for (let z = 0; z < this.gridDim.z; ++z) {
      for (let x = 0; x < this.gridDim.x; ++x) {

        let currIdx = z * this.gridDim.x + x;

        let tlIdx = currIdx + this.gridDim.x - 1;
        let  tIdx = currIdx + this.gridDim.x;
        let trIdx = currIdx + this.gridDim.x + 1;
        let  rIdx = currIdx + 1;
        let brIdx = currIdx - this.gridDim.x + 1;
        let  bIdx = currIdx - this.gridDim.x;
        let blIdx = currIdx - this.gridDim.x - 1;
        let  lIdx = currIdx - 1;

        let vertice = this.vertices[currIdx];

        if (x > 0) {
          vertice.neighbors['l'] = this.vertices[lIdx];
          if (z > 0) {
            vertice.neighbors['bl'] = this.vertices[blIdx];
          }
        }
        if (z > 0) {
          vertice.neighbors['b'] = this.vertices[bIdx];
          if (x < this.gridDim.x - 1) {
            vertice.neighbors['br'] = this.vertices[brIdx];
          }
        }
        if (x < this.gridDim.x - 1) {
          vertice.neighbors['r'] = this.vertices[rIdx];
          if (z < this.gridDim.z - 1) {
            vertice.neighbors['tr'] = this.vertices[trIdx];
          }
        }
        if (z < this.gridDim.z - 1) {
          vertice.neighbors['t'] = this.vertices[tIdx];
          if (x > 0) {
            vertice.neighbors['tl'] = this.vertices[tlIdx];
          }
        }

        if (x < this.gridDim.x - 1 && z < this.gridDim.z - 1) {
          let face1 = new THREE.Face3(currIdx, tIdx, trIdx);
          let face2 = new THREE.Face3(currIdx, trIdx, rIdx);
          face1.color = new THREE.Color(0x0000ff);
          face2.color = new THREE.Color(0xff0000);
          vertice.faces['a'] = face1;
          vertice.faces['b'] = face2;
          geometryMesh.faces.push(face1);
          geometryMesh.faces.push(face2);
        }
      }
    }

    this.mesh = new THREE.Mesh(geometryMesh, new THREE.MeshBasicMaterial({
      vertexColors: THREE.FaceColors
    }));

    geometryPoints.computeBoundingSphere();

    this.points = new THREE.Points(geometryPoints, new THREE.PointsMaterial({
        size: pointSize,
        color: 0xff0000
    }));

  }

  update(occupancyGrid) {
    this.vertices.forEach(vertice => {
      vertice.updatePosition(occupancyGrid);
      vertice.updateColor(occupancyGrid);
    });
    // this.mesh.geometry.elementsNeedUpdate = true;
    this.mesh.geometry.colorsNeedUpdate = true;
    this.mesh.geometry.verticesNeedUpdate = true;
    this.points.geometry.verticesNeedUpdate = true;
  }

}
