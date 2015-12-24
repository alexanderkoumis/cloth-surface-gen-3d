'use strict'

class OccupancyGrid {

  constructor(points, blockDim) {

    function calcGridDim(boundingBox, blockDim) {
      let bBoxSize = boundingBox.max.clone().sub(boundingBox.min);
      return new THREE.Vector3(
        parseInt(bBoxSize.x / blockDim.x),
        parseInt(bBoxSize.y / blockDim.y),
        parseInt(bBoxSize.z / blockDim.z)
      );
    }

    function calcGridPos(position, boundingBox, blockDim, gridDim, numGridElements) {
      let offsetPosition = position.clone().sub(boundingBox.min);
      let gridX = offsetPosition.x / blockDim.x;
      let gridY = offsetPosition.y / blockDim.y;
      let gridZ = offsetPosition.z / blockDim.z;
      let flatIdx = OccupancyGrid.calcFlatIdx(gridX, gridY, gridZ, gridDim);
      return Math.min(flatIdx, numGridElements - 1);
    }

    this.boundingBox = this.computeBoundingBox(points);
    this.blockDim = blockDim;
    this.gridDim = calcGridDim(this.boundingBox, blockDim);

    const numGridElements = this.gridDim.x * this.gridDim.y * this.gridDim.z;

    this.elements = [].slice.apply(new Int32Array(numGridElements));
    this.occupancy = [].slice.apply(new Float32Array(numGridElements)); 


    points.geometry.vertices.forEach(vertice => {
      const gridPosition = calcGridPos(vertice, this.boundingBox,
                                       blockDim, this.gridDim, numGridElements);
      this.elements[gridPosition]++;
    });

    let maxPts = 0;
    this.elements.forEach(numPts => {
      maxPts = Math.max(numPts, maxPts);
    });

    for (let i = 0; i < this.elements.length; ++i) {
      this.occupancy[i] = this.elements[i] / maxPts;
    }

  }

  static calcFlatIdx(x, y, z, gridDim) {
    return parseInt(z) * gridDim.x * gridDim.y +
           parseInt(y) * gridDim.x +
           parseInt(x);
  }

  line(vStart, vEnd, color, lineWidth) {
    var material = new THREE.LineBasicMaterial({
      color: color,
      linewidth: lineWidth
    });
    var geometry = new THREE.Geometry();
    geometry.vertices.push(vStart);
    geometry.vertices.push(vEnd);
    return new THREE.Line(geometry, material);
  }


  drawBox(scene, min, max, color, lineWidth) {
    scene.add(this.line(new THREE.Vector3(min.x, min.y, min.z), new THREE.Vector3(max.x, min.y, min.z), color, lineWidth));
    scene.add(this.line(new THREE.Vector3(min.x, min.y, max.z), new THREE.Vector3(max.x, min.y, max.z), color, lineWidth));
    scene.add(this.line(new THREE.Vector3(min.x, max.y, min.z), new THREE.Vector3(max.x, max.y, min.z), color, lineWidth));
    scene.add(this.line(new THREE.Vector3(min.x, max.y, max.z), new THREE.Vector3(max.x, max.y, max.z), color, lineWidth));
    scene.add(this.line(new THREE.Vector3(min.x, min.y, min.z), new THREE.Vector3(min.x, max.y, min.z), color, lineWidth));
    scene.add(this.line(new THREE.Vector3(min.x, min.y, max.z), new THREE.Vector3(min.x, max.y, max.z), color, lineWidth));
    scene.add(this.line(new THREE.Vector3(max.x, min.y, min.z), new THREE.Vector3(max.x, max.y, min.z), color, lineWidth));
    scene.add(this.line(new THREE.Vector3(max.x, min.y, max.z), new THREE.Vector3(max.x, max.y, max.z), color, lineWidth));
    scene.add(this.line(new THREE.Vector3(min.x, min.y, min.z), new THREE.Vector3(min.x, min.y, max.z), color, lineWidth));
    scene.add(this.line(new THREE.Vector3(min.x, max.y, min.z), new THREE.Vector3(min.x, max.y, max.z), color, lineWidth));
    scene.add(this.line(new THREE.Vector3(max.x, min.y, min.z), new THREE.Vector3(max.x, min.y, max.z), color, lineWidth));
    scene.add(this.line(new THREE.Vector3(max.x, max.y, min.z), new THREE.Vector3(max.x, max.y, max.z), color, lineWidth));
  }

  drawBoundingBox(scene) {
    this.drawBox(scene, this.boundingBox.min, this.boundingBox.max, 0x00ff00, 5);
  }

  drawGrid(scene) {
    for (let x = 0; x < this.gridDim.x; ++x) {
      for (let y = 0; y < this.gridDim.y; ++y) {
        for (let z = 0; z < this.gridDim.z; ++z) {

          let positionBase = this.boundingBox.min.clone();
          let positionMin = positionBase.add(new THREE.Vector3(
            x * this.blockDim.x,
            y * this.blockDim.y,
            z * this.blockDim.z
          ));
          let positionMax = positionMin.clone().add(this.blockDim);
          // this.drawBox(scene, positionMin, positionMax, 0x0000ff, 2);
          let flatIdx = OccupancyGrid.calcFlatIdx(x, y, z, this.gridDim);
          let occupancy = this.occupancy[flatIdx].toPrecision(3);
          let elements = this.elements[flatIdx].toPrecision(3);
          let textShapes = THREE.FontUtils.generateShapes(String(occupancy), {
            'font': 'helvetiker',
            'weight': 'normal',
            'style': 'normal',
            'size': 0.05,
            'curveSegments': 10
          });
          let textMesh = new THREE.Mesh(
            new THREE.ShapeGeometry(textShapes),
            new THREE.MeshBasicMaterial({
              color: 0x0000ff,
              side: THREE.DoubleSide,
              transparent: true,
              opacity: occupancy
            })
          );
          scene.add(textMesh);
          textMesh.position.set(
            positionMin.x + this.blockDim.x / 2,
            positionMin.y + this.blockDim.y / 2,
            positionMin.z + this.blockDim.z / 2
          );
        }
      }
    }
  }

  computeBoundingBox(points) {
    let min = new THREE.Vector3();
    let max = new THREE.Vector3();
    points.geometry.vertices.forEach(vertice => {
      min.x = Math.min(vertice.x, min.x);
      min.y = Math.min(vertice.y, min.y);
      min.z = Math.min(vertice.z, min.z);
      max.x = Math.max(vertice.x, max.x);
      max.y = Math.max(vertice.y, max.y);
      max.z = Math.max(vertice.z, max.z);
    });
    return {
      min: min,
      max: max
    }
  }

}
