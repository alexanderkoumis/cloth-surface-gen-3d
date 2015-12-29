'use strict'

class Vertice {

  constructor(position, moveVec, fromSplit, minFillRatio) {
    this.position = position;
    this.nextPosition = position;
    this.moveVec = moveVec;
    this.fromSplit = fromSplit;
    this.minFillRatio = minFillRatio;
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
    };
  }

  updatePosition(occupancyGrid) {
    this.nextPosition = this.position.clone().add(this.moveVec);
    let nextGridPosition = occupancyGrid.calcGridPos(this.nextPosition);
    let ratioFilled = occupancyGrid.getRatioFilled(nextGridPosition);
    if (ratioFilled < this.minFillRatio) {
      this.position.add(this.moveVec);
    }
  }

  updateColor(occupancyGrid) {
    if (this.faces.a && this.faces.b) {
      let newColor = occupancyGrid.getColor(this.nextPosition);
      if (newColor.r > 0 && newColor.g > 0 && newColor.b > 0) {
        this.faces.a.color.set(newColor);
        this.faces.b.color.set(newColor);
      }
    }
  }

  // calcNeighborDistances() {
  //   let back = this.neighbors.back;
  //   let front = this.neighbors.front;    
  //   return {
  //     back: back ? this.position.distanceTo(back.position) : 0.0,
  //     front: front ? this.position.distanceTo(front.position) : 0.0
  //   }
  // }

  // calcNewPosition(grid) {
  //   const positionToTry = this.position.clone().add(this.moveVec);
  //   const fillRatio = grid.calcFillRatio(positionToTry);
  //   if (fillRatio > this.minFillRatio) {
  //     return this.position.clone();
  //   }
  //   return positionToTry;
  // }

  // genNewPoint(grid) {
  //   return new Point(
  //     this.calcNewPosition(grid),
  //     this.spacing,
  //     this.radius,
  //     this.moveVec,
  //     this.minFillRatio,
  //     false
  //   );
  // }

  // genSplitPoints(ptSplitMultiple) {
  //   let splitPoints = [];
  //   let neighborDistances = this.calcNeighborDistances();
  //   if (neighborDistances.front > this.spacing * ptSplitMultiple) {
  //     let positionFront = this.neighbors.front.position;
  //     const deltaY = positionFront.y - this.position.y;
  //     const deltaX = positionFront.x - this.position.x;
  //     const angleOrigRad = Math.atan2(deltaY, deltaX);
  //     const angleSpacingRad = Math.PI / ptSplitMultiple;
  //     const position = new THREE.Vector2(
  //       this.position.x + deltaX / 2,
  //       this.position.y + deltaY / 2
  //     );
  //     for (let i = 1; i < ptSplitMultiple; ++i) {
  //       const angleSplitRad = angleOrigRad + i * angleSpacingRad;
  //       const moveVec = new THREE.Vector2(
  //         Math.abs(this.moveVec.y) * Math.cos(angleSplitRad),
  //         Math.abs(this.moveVec.y) * Math.sin(angleSplitRad)
  //       );
  //       let splitPoint = new Point(
  //         position,
  //         this.spacing,
  //         this.radius,
  //         moveVec,
  //         this.minFillRatio,
  //         true
  //       );
  //       splitPoints.push(splitPoint);
  //     }
  //   }
  //   return splitPoints.reverse();
  // }

  // static computeAngle(pointBack, pointCurr, pointFront) {
  //   let back = pointBack.position;
  //   let curr = pointCurr.position;
  //   let front = pointFront.position;
  //   const angle1 = Math.atan2(curr.y - back.y, curr.x - back.x);
  //   const angle2 = Math.atan2(curr.y - front.y, curr.x - front.x);
  //   return angle1 - angle2;
  // }

  // static setNeighbors(points) {
  //   const numPoints = points.length;
  //   for (let i = 0; i < numPoints; ++i) {
  //     const hasBack = i > 0;
  //     const hasFront = i < (numPoints - 1); 
  //     points[i].neighbors = {
  //       back : hasBack  ? points[i - 1] : null,
  //       front: hasFront ? points[i + 1] : null,
  //       angle: (hasBack && hasFront) ? Point.computeAngle(points[i - 1],
  //                                                         points[i    ],
  //                                                         points[i + 1]) : 0.0
  //     }
  //   }
  // }

  // static genInitPoints(numPts, width, height, radius, moveVec, minFillRatio) {
  //   let points = [];
  //   const spacing = width / numPts;
  //   for (let i = 0; i < numPts; ++i) {
  //     let position = new THREE.Vector2(i * spacing, 5);
  //     points.push(new Point(position, spacing, radius, moveVec, minFillRatio,
  //                           false));
  //   }
  //   Point.setNeighbors(points);
  //   return points;
  // }

  // static genNewPoints(points, grid, ptSplitMultiple) {
  //   let newPoints = [];
  //   let tempPointPairs = [];
  //   for (let i = 0; i < points.length; ++i) {
  //     let point = points[i];
  //     let newPoint = point.genNewPoint(grid);
  //     newPoints.push(newPoint);
  //     tempPointPairs.push({ oldIdx: i, newIdx: newPoints.length - 1 });
  //     point.genSplitPoints(ptSplitMultiple).forEach(splitPoint => {
  //       newPoints.push(splitPoint);
  //     });
  //   };
  //   Point.setNeighbors(newPoints);
  //   tempPointPairs.forEach(pointPair => {
  //     let oldPoint = points[pointPair.oldIdx];
  //     let newPoint = newPoints[pointPair.newIdx];
  //     if (oldPoint && newPoint && newPoint.fromSplit) {
  //       if (oldPoint.neighbors && newPoint.neighbors) {
  //         const oldAngle = oldPoint.neighbors.angle;
  //         const newAngle = newPoint.neighbors.angle;
  //         if ((oldAngle > 0.0 && newAngle < 0.0) ||
  //             (oldAngle < 0.0 && newAngle > 0.0)) {
  //           newPoints.splice(pointPair.newIdx - 1, 3, newPoint);
  //         }
  //       }
  //     }
  //   });
  //   Point.setNeighbors(newPoints);
  //   return newPoints;
  // }

  // static draw(context, points) {
  //   context.fillStyle = 'green';
  //   points.forEach(point => {
  //     context.beginPath();
  //     context.arc(
  //       point.position.x,
  //       point.position.y,
  //       point.radius, 0, Math.PI*2, true);
  //     context.fill();
  //   });
  // }

}
