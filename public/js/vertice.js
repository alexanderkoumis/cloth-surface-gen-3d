'use strict'

class Vertice {

  constructor(position, moveVec, minFillRatio, maxNeighborDist, fromSplit, idx) {
    this.position = position;
    this.nextPosition = position;
    this.moveVec = moveVec;
    this.minFillRatio = minFillRatio;
    this.maxNeighborDist = maxNeighborDist;
    this.fromSplit = fromSplit;
    this.idx = idx;
    this.done = false;
    this.faces = {
      a: null,
      b: null
    }
    this.neighbors = {
      tl: null,
      t:  null,
      tr: null,
      r:  null,
      br: null,
      b:  null,
      bl: null,
      l:  null
    }
  }

  neighborDistance(neighbor) {
    if (this.neighbors[neighbor]) {
      let dist = this.position.distanceTo(this.neighbors[neighbor].position);
      if (!isNaN(dist)) {
        return dist;
      }
    }
    return 0.0;
  }

  updatePosition(occupancyGrid) {
    if (this.done) {
      return;
    }
    this.nextPosition = this.position.clone().add(this.moveVec);
    let nextGridPosition = occupancyGrid.calcGridPos(this.nextPosition);
    let ratioFilled = occupancyGrid.getRatioFilled(nextGridPosition);
    if (ratioFilled < this.minFillRatio) {
      if (this.nextPosition.z > occupancyGrid.bbox.min.z) {
        if ((this.neighborDistance('tl') > this.maxNeighborDist) ||
            (this.neighborDistance('t')  > this.maxNeighborDist) ||
            (this.neighborDistance('tr') > this.maxNeighborDist) ||
            (this.neighborDistance('r')  > this.maxNeighborDist) ||
            (this.neighborDistance('br') > this.maxNeighborDist) ||
            (this.neighborDistance('b')  > this.maxNeighborDist) ||
            (this.neighborDistance('bl') > this.maxNeighborDist) ||
            (this.neighborDistance('l')  > this.maxNeighborDist)) {
          return;
        }
        this.position.add(this.moveVec);
      }
      else {
        this.done = true;
      }
    }
  }

  updateColor(occupancyGrid) {
    if (this.done) {
      return;
    }
    if (this.faces.a && this.faces.b) {
      let newColor = occupancyGrid.getColor(this.nextPosition);
      if (newColor.r > 0.0 && newColor.g > 0.0 && newColor.b > 0.0) {
        this.faces.a.color.set(newColor);
        this.faces.b.color.set(newColor);
      }
    }
  }

}
