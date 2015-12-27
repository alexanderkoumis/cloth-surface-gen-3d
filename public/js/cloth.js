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

    let geometry = new THREE.Geometry();
    for (let z = 0; z < this.gridDim.z; ++z) {
      for (let x = 0; x < this.gridDim.x; ++x) {
        let position = new THREE.Vector3(x * blockDim.x + boundingBox.min.x,
                                        boundingBox.max.y,
                                        z * blockDim.z + boundingBox.min.z);
        let moveVec = new THREE.Vector3(0, -moveScalar, 0);
        let fromSplit = false;
        geometry.vertices.push(position);
        this.vertices.push(new Vertice(position, moveVec, fromSplit,
                                       minFillRatio));
      }
    }

    this.setNeighbors();

    geometry.computeBoundingSphere();
    this.points = new THREE.Points(geometry, new THREE.PointsMaterial({
        size: pointSize,
        color: 0xff0000
    }));



  }

  update(occupancyGrid) {
    this.vertices.forEach(vertice => {
      vertice.updatePosition(occupancyGrid);
    });
    this.points.geometry.verticesNeedUpdate = true;
  }

  setNeighbors() {
    // for (let z = 0; z < this.gridDim.z; ++z) {
    //   for (let x = )
    // }
  }

  drawCloth(scene) {

  }


  static calcFlatIdx(x, y, clothDim) {
    return parseInt(y) * clothDim.x + parseInt(x);
  }

}
