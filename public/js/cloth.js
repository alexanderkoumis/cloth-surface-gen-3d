'use strict'

class Cloth {

  constructor(blockDim, boundingBox, moveScalar, pointSize) {

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
    this.vertices = this.genInitVertices(blockDim, this.gridDim, boundingBox,
                                         moveScalar);
  }

  genInitVertices(blockDim, gridDim, boundingBox, moveScalar) {
    let vertices = [];
    for (let z = 0; z < gridDim.z; ++z) {
      for (let x = 0; x < gridDim.x; ++x) {
        let position = new THREE.Vector3(x * blockDim.x + boundingBox.min.x,
                                        boundingBox.max.y,
                                        z * blockDim.z + boundingBox.min.z);
        let moveVec = new THREE.Vector3(0, 0, -moveScalar);
        let fromSplit = false;
        vertices.push(new Vertice(position, moveVec, fromSplit));
      }
    }
    return vertices;
  }

  genPoints() {
    let geometry = new THREE.BufferGeometry();
    let material = new THREE.PointsMaterial({
        size: this.pointSize,
        color: 0xff0000
    });
    let positions = new Float32Array(this.vertices.length * 3);
    for (let i = 0, j = 0; i < this.vertices.length; i += 1, j += 3) {
      positions[j + 0] = this.vertices[i].position.x;
      positions[j + 1] = this.vertices[i].position.y;
      positions[j + 2] = this.vertices[i].position.z;
    };
    geometry.addAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.computeBoundingBox();
    let points = new THREE.Points(geometry, material);
    return points;
  }

  static calcFlatIdx(x, y, clothDim) {
    return parseInt(y) * clothDim.x + parseInt(x);
  }

}
