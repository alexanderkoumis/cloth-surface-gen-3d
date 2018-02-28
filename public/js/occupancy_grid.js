'use strict'

class OccupancyGrid {

  constructor(pointCloud, blockDim, bboxLineWidth) {

    function calcGridDim(boundingBox, blockDim) {
      let bBoxSize = boundingBox.max.clone().sub(boundingBox.min);
      return new THREE.Vector3(
        parseInt(bBoxSize.x / blockDim.x),
        parseInt(bBoxSize.y / blockDim.y),
        parseInt(bBoxSize.z / blockDim.z)
      );
    }

    const pointPositions = pointCloud.geometry.attributes.position.array;
    const pointColors = pointCloud.geometry.attributes.color.array;

    this.bbox = this.computeBoundingBox(pointPositions);
    this.bboxObj = this.createBoxLine(this.bbox.min, this.bbox.max, 0x00ff00, 1,
                                      bboxLineWidth);
    this.blockDim = blockDim;
    this.gridDim = calcGridDim(this.bbox, blockDim);

    this.numGridElements = this.gridDim.x * this.gridDim.y * this.gridDim.z;

    this.elements = [].slice.apply(new Int32Array(this.numGridElements));
    this.occupancy = [].slice.apply(new Float32Array(this.numGridElements));
    this.colors = [].slice.apply(new Float32Array(this.numGridElements * 3));

    for (let i = 0; i < pointPositions.length; i += 3) {
      let verticeX = pointPositions[i];
      let verticeY = pointPositions[i+1];
      let verticeZ = pointPositions[i+2];
      let colorR = pointColors[i];
      let colorG = pointColors[i+1];
      let colorB = pointColors[i+2];
      const gridPosition = this.calcGridPos(verticeX, verticeY, verticeZ);
      const gridPositionColor = gridPosition * 3;
      this.elements[gridPosition] += 1;
      this.colors[gridPositionColor + 0] += colorR;
      this.colors[gridPositionColor + 1] += colorG;
      this.colors[gridPositionColor + 2] += colorB;
    }

    for (let i = 0, j = 0; i < this.elements.length; i += 1, j += 3) {
      let numElements = this.elements[i];
      if (numElements > 0) {
        this.colors[j + 0] /= numElements;
        this.colors[j + 1] /= numElements;
        this.colors[j + 2] /= numElements;
      }
    }

    let maxPts = 0;
    this.elements.forEach(numPts => {
      maxPts = Math.max(numPts, maxPts);
    });

    for (let i = 0; i < this.elements.length; ++i) {
      this.occupancy[i] = this.elements[i] / maxPts;
    }

  }

  calcFlatIdx(x, y, z) {
    return parseInt(z) * this.gridDim.y * this.gridDim.x+
           parseInt(y) * this.gridDim.x +
           parseInt(x);
  }

  calcGridPos(x, y, z, log) {
    let offsetX = x - this.bbox.min.x;
    let offsetY = y - this.bbox.min.y;
    let offsetZ = z - this.bbox.min.z;
    let gridX = offsetX / this.blockDim.x;
    let gridY = offsetY / this.blockDim.y;
    let gridZ = offsetZ / this.blockDim.z;
    let flatIdx = this.calcFlatIdx(gridX, gridY, gridZ);
    if (log) {
      console.log('[', gridX, gridY, gridZ, ']', flatIdx, this.numGridElements);
    }
    return Math.min(flatIdx, this.numGridElements - 1);
  }

  getColor(position) {
    let gridPosition = this.calcGridPos(position.x, position.y, position.z);
    const gridPositionColor = gridPosition * 3;
    return new THREE.Color(
      this.colors[gridPositionColor    ],
      this.colors[gridPositionColor + 1],
      this.colors[gridPositionColor + 2]
    );
  }

  getRatioFilled(gridPosition) {
    return this.occupancy[gridPosition];
  }

  createBoxLine(min, max, color, opacity, lineWidth) {
    var geometry = new THREE.Geometry();
    geometry.vertices.push(new THREE.Vector3(min.x, min.y, min.z));
    geometry.vertices.push(new THREE.Vector3(max.x, min.y, min.z));
    geometry.vertices.push(new THREE.Vector3(max.x, min.y, max.z));
    geometry.vertices.push(new THREE.Vector3(min.x, min.y, max.z));
    geometry.vertices.push(new THREE.Vector3(min.x, min.y, min.z));
    geometry.vertices.push(new THREE.Vector3(min.x, max.y, min.z));
    geometry.vertices.push(new THREE.Vector3(max.x, max.y, min.z));
    geometry.vertices.push(new THREE.Vector3(max.x, max.y, max.z));
    geometry.vertices.push(new THREE.Vector3(min.x, max.y, max.z));
    geometry.vertices.push(new THREE.Vector3(min.x, max.y, min.z));
    geometry.vertices.push(new THREE.Vector3(max.x, max.y, min.z));
    geometry.vertices.push(new THREE.Vector3(max.x, min.y, min.z));
    geometry.vertices.push(new THREE.Vector3(max.x, min.y, max.z));
    geometry.vertices.push(new THREE.Vector3(max.x, max.y, max.z));
    geometry.vertices.push(new THREE.Vector3(min.x, max.y, max.z));
    geometry.vertices.push(new THREE.Vector3(min.x, min.y, max.z));
    return new THREE.Line(geometry, new THREE.LineBasicMaterial({
      color: color,
      linewidth: lineWidth,
      transparent: true,
      opacity: opacity
    }));
  }

  drawGrid(scene) {
    for (let x = 0; x < this.gridDim.x; ++x) {
      for (let y = 0; y < this.gridDim.y; ++y) {
        for (let z = 0; z < this.gridDim.z; ++z) {
          let positionBase = this.bbox.min.clone();
          let positionMin = positionBase.add(new THREE.Vector3(
            x * this.blockDim.x,
            y * this.blockDim.y,
            z * this.blockDim.z
          ));
          let position = positionMin.clone().add(this.blockDim.clone().divideScalar(2));
          let positionMax = positionMin.clone().add(this.blockDim);
          let color = this.getColor(position);
          let flatIdx = this.calcFlatIdx(x, y, z);
          let occupancy = this.occupancy[flatIdx].toPrecision(3);
          let boxLine = this.createBoxLine(positionMin, positionMax, color.getHex(), occupancy, 2);
          scene.add(boxLine);
          // let elements = this.elements[flatIdx].toPrecision(3);
          // let textShapes = THREE.FontUtils.generateShapes(String(occupancy), {
          //   'font': 'helvetiker',
          //   'weight': 'normal',
          //   'style': 'normal',
          //   'size': 0.05,
          //   'curveSegments': 10
          // });
          // let textMesh = new THREE.Mesh(
          //   new THREE.ShapeGeometry(textShapes),
          //   new THREE.MeshBasicMaterial({
          //     color: 0x0000ff,
          //     side: THREE.DoubleSide,
          //     transparent: true,
          //     opacity: occupancy
          //   })
          // );
          // scene.add(textMesh);
          // textMesh.position.set(
          //   positionMin.x + this.blockDim.x / 2,
          //   positionMin.y + this.blockDim.y / 2,
          //   positionMin.z + this.blockDim.z / 2
          // );
        }
      }
    }
  }

  computeBoundingBox(pointPositions, bboxLineWidth) {
    let min = new THREE.Vector3();
    let max = new THREE.Vector3();
    for (let i = 0; i < pointPositions.length; i += 3) {
      let verticeX = pointPositions[i];
      let verticeY = pointPositions[i+1];
      let verticeZ = pointPositions[i+2];
      min.x = Math.min(verticeX, min.x);
      min.y = Math.min(verticeY, min.y);
      min.z = Math.min(verticeZ, min.z);
      max.x = Math.max(verticeX, max.x);
      max.y = Math.max(verticeY, max.y);
      max.z = Math.max(verticeZ, max.z);
    }
    return {
      min: min,
      max: max
    }
  }

}
