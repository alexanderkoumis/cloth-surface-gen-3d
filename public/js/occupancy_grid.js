'use strict'

class OccupancyGrid {

  constructor(points, blockDim, bboxLineWidth) {

    function calcGridDim(boundingBox, blockDim) {
      let bBoxSize = boundingBox.max.clone().sub(boundingBox.min);
      return new THREE.Vector3(
        parseInt(bBoxSize.x / blockDim.x),
        parseInt(bBoxSize.y / blockDim.y),
        parseInt(bBoxSize.z / blockDim.z)
      );
    }

    this.bbox = this.computeBoundingBox(points);
    this.bboxObj = this.createBoxLine(this.bbox.min, this.bbox.max, 0x00ff00,
                                      bboxLineWidth);
    this.blockDim = blockDim;
    this.gridDim = calcGridDim(this.bbox, blockDim);

    this.numGridElements = this.gridDim.x * this.gridDim.y * this.gridDim.z;

    this.elements = [].slice.apply(new Int32Array(this.numGridElements));
    this.occupancy = [].slice.apply(new Float32Array(this.numGridElements));
    this.colors = [].slice.apply(new Float32Array(this.numGridElements * 3));

    for (let i = 0, j = 0; i < points.geometry.vertices.length; i += 1, j += 3) {
      let vertice = points.geometry.vertices[i];
      let color = points.geometry.colors[i];
      const gridPosition = this.calcGridPos(vertice);
      const gridPositionColor = gridPosition * 3;
      this.elements[gridPosition]++;
      this.colors[gridPositionColor + 0] += color.r;
      this.colors[gridPositionColor + 1] += color.g;
      this.colors[gridPositionColor + 2] += color.b;
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
    return parseInt(z) * this.gridDim.x * this.gridDim.y +
           parseInt(y) * this.gridDim.x +
           parseInt(x);
  }

  calcGridPos(position) {
    let offsetPosition = position.clone().sub(this.bbox.min);
    let gridX = offsetPosition.x / this.blockDim.x;
    let gridY = offsetPosition.y / this.blockDim.y;
    let gridZ = offsetPosition.z / this.blockDim.z;
    let flatIdx = this.calcFlatIdx(gridX, gridY, gridZ);
    return Math.min(flatIdx, this.numGridElements - 1);
  }

  getColor(position) {
    let gridPosition = this.calcGridPos(position);
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

  createBoxLine(min, max, color, lineWidth) {
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
      linewidth: lineWidth
    }));
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
          let boxLine = this.createBoxLine(positionMin, positionMax, 0x0000ff,
                                           2);
          scene.add(boxLine);
          let flatIdx = this.calcFlatIdx(x, y, z);
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

  computeBoundingBox(points, bboxLineWidth) {
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
