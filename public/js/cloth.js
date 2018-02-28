  'use strict'

class Cloth {

  constructor(pointCloud, optionsUser) {

    function calcGridDim(boundingBox, blockDim) {
      let bBoxSize = boundingBox.max.clone().sub(boundingBox.min);
      return new THREE.Vector3(
        parseInt(bBoxSize.x / blockDim.x),
        parseInt(bBoxSize.y / blockDim.y),
        1
      );
    }

    let optionsDefault = {
      bboxLineWidth: 2,
      minFillRatio: 0.01,
      maxNeighborDist: 0.1,
      pointSize: 0.03,
      moveVec: new THREE.Vector3(0, 0, -0.01),
      blockDimGrid: new THREE.Vector3(0.03, 0.03, 0.03),
      blockDimCloth: new THREE.Vector3(0.04, 0.04, 0.03)
    }

    let options = {};

    Object.keys(optionsDefault).forEach(function(option) {
      options[option] = optionsUser   [option] === undefined ?
                        optionsDefault[option] :
                        optionsUser   [option] ;
    });

    this.occupancyGrid = new OccupancyGrid(pointCloud,
                                           options.blockDimGrid,
                                           options.bboxLineWidth);

    let bbox = this.occupancyGrid.bbox;

    this.blockDim = options.blockDimCloth;
    this.gridDim = calcGridDim(bbox, options.blockDimCloth);
    this.boundingBox = bbox;
    this.moveVec = options.moveVec;
    this.pointSize = options.pointSize;
    this.minFillRatio = options.minFillRatio;
    this.maxNeighborDist = options.maxNeighborDist;
    this.vertices = [];
    let geometryPoints = new THREE.Geometry();
    for (let y = 0; y < this.gridDim.y; ++y) {
      for (let x = 0; x < this.gridDim.x; ++x) {
        let position = new THREE.Vector3(x * this.blockDim.x + bbox.min.x,
                                         y * this.blockDim.y + bbox.min.y,
                                         bbox.max.z);
        let idx = y * this.gridDim.x + x;
        geometryPoints.vertices.push(position);
        this.vertices.push(new Vertice(position, this.moveVec, this.minFillRatio,
                                       this.maxNeighborDist, false, idx));
      }
    }

    let geometryMesh = new THREE.Geometry();
    geometryMesh.vertices = geometryPoints.vertices;

    function checkIdxs(idx1, idx2) {
      if (idx1 != idx2) {
        console.log('idx1', idx1, 'idx2', idx2);
        debugger;
      }
    }

    for (let y = 0; y < this.gridDim.y; ++y) {
      for (let x = 0; x < this.gridDim.x; ++x) {

        let currIdx = y * this.gridDim.x + x;

        let  lIdx = currIdx - 1;
        let tlIdx = currIdx - this.gridDim.x - 1;
        let  tIdx = currIdx - this.gridDim.x;
        let trIdx = currIdx - this.gridDim.x + 1;
        let  rIdx = currIdx + 1;
        let brIdx = currIdx + this.gridDim.x + 1;
        let  bIdx = currIdx + this.gridDim.x;
        let blIdx = currIdx + this.gridDim.x - 1;

        let vertice = this.vertices[currIdx];

        if (x > 0) {
          vertice.neighbors['l'] = this.vertices[lIdx];
          if (y > 0) {
            vertice.neighbors['tl'] = this.vertices[tlIdx];
          }
        }
        if (y > 0) {
          vertice.neighbors['t'] = this.vertices[tIdx];
          if (x < this.gridDim.x - 1) {
            vertice.neighbors['tr'] = this.vertices[trIdx];
          }
        }
        if (x < this.gridDim.x - 1) {
          vertice.neighbors['r'] = this.vertices[rIdx];
          if (y < this.gridDim.y - 1) {
            vertice.neighbors['br'] = this.vertices[brIdx];
          }
        }
        if (y < this.gridDim.y - 1) {
          vertice.neighbors['b'] = this.vertices[bIdx];
          if (x > 0) {
            vertice.neighbors['bl'] = this.vertices[blIdx];
          }
        }

        if (x < this.gridDim.x - 1 && y < this.gridDim.y - 1) {
          let face1 = new THREE.Face3(currIdx, rIdx, brIdx);
          let face2 = new THREE.Face3(currIdx, brIdx, bIdx);
          face1.color = new THREE.Color(0x0000ff);
          face2.color = new THREE.Color(0xff0000);
          vertice.faces['a'] = face1;
          vertice.faces['b'] = face2;
          geometryMesh.faces.push(face1);
          geometryMesh.faces.push(face2);
        }
      }
    }

    let material = new THREE.MeshBasicMaterial({ vertexColors: THREE.FaceColors });
    this.mesh = new THREE.Mesh(geometryMesh, material);

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
    this.mesh.geometry.elementsNeedUpdate = true;
    this.mesh.geometry.colorsNeedUpdate = true;
    this.mesh.geometry.verticesNeedUpdate = true;
    this.points.geometry.verticesNeedUpdate = true;
  }

}
