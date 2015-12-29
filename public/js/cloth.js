  'use strict'

class Cloth {

  constructor(pointCloud, optionsUser) {

    function calcGridDim(boundingBox, blockDim) {
      let bBoxSize = boundingBox.max.clone().sub(boundingBox.min);
      return new THREE.Vector3(
        parseInt(bBoxSize.x / blockDim.x),
        1,
        parseInt(bBoxSize.z / blockDim.z)
      );
    }

    let optionsDefault = {
      bboxLineWidth: 2,
      minFillRatio: 0.2,
      maxNeighborDist: 0.1,
      moveScalar: 0.01,
      pointSize: 0.03,
      blockDimGrid: new THREE.Vector3(0.03, 0.03, 0.03),
      blockDimCloth: new THREE.Vector3(0.04, 0.04, 0.03)
    }

    let options = {};

    Object.keys(optionsDefault).forEach(function(option) {
      options[option] = optionsUser[option] === undefined ?
                        optionsDefault[option] :
                        optionsUser[option];
    });

    this.occupancyGrid = new OccupancyGrid(pointCloud, options.blockDimGrid,
                                           options.bboxLineWidth);

    let bbox = this.occupancyGrid.bbox;

    this.blockDim = options.blockDimCloth;
    this.gridDim = calcGridDim(bbox, options.blockDimCloth);
    this.boundingBox = bbox;
    this.moveScalar = options.moveScalar;
    this.pointSize = options.pointSize;
    this.minFillRatio = options.minFillRatio;
    this.maxNeighborDist = options.maxNeighborDist;
    this.vertices = [];

    let geometryPoints = new THREE.Geometry();
    for (let z = 0; z < this.gridDim.z; ++z) {
      for (let x = 0; x < this.gridDim.x; ++x) {
        let position = new THREE.Vector3(x * this.blockDim.x + bbox.min.x,
                                        bbox.max.y,
                                        z * this.blockDim.z + bbox.min.z);
        let moveVec = new THREE.Vector3(0, -this.moveScalar, 0);
        let fromSplit = false;
        geometryPoints.vertices.push(position);
        this.vertices.push(new Vertice(position, moveVec, this.minFillRatio,
                                       this.maxNeighborDist, fromSplit));
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

    this.mesh = new THREE.Mesh(geometryMesh, new THREE.MeshFaceMaterial([
      new THREE.MeshBasicMaterial({ opacity: 1, vertexColors: THREE.FaceColors }),
      new THREE.MeshBasicMaterial({ opacity: 0, vertexColors: THREE.FaceColors })
    ]));

    geometryPoints.computeBoundingSphere();

    this.points = new THREE.Points(geometryPoints, new THREE.PointsMaterial({
        size: this.pointSize,
        color: 0xff0000
    }));

  }

  update() {
    this.vertices.forEach(vertice => {
      vertice.updatePosition(this.occupancyGrid);
      vertice.updateColor(this.occupancyGrid);
    });
    this.mesh.geometry.colorsNeedUpdate = true;
    this.mesh.geometry.verticesNeedUpdate = true;
    this.points.geometry.verticesNeedUpdate = true;
  }

}
