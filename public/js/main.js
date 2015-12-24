'use strict'

function computeCentroid(points) {
  let centroid = new THREE.Vector3();
  const vertices = points.geometry.vertices;
  points.geometry.vertices.forEach(vertice => {
    centroid.add(vertice);
  });
  centroid.divideScalar(vertices.length);
  return centroid;
}

let points = null;
let burritoRotation = new THREE.Euler(2.1, 0.0584, 2.0584)

function loadPoints(loadedCallback) {
  new THREE.PLYLoader().load('ply/burrito_trimmed.ply', function (geometry) {
    points = new THREE.Points(geometry, new THREE.PointsMaterial({
      size: 0.1,
      vertexColors: THREE.VertexColors
    }));
    const centroid = computeCentroid(points);
    points.geometry.vertices.forEach(vertice => {
      vertice.sub(centroid);
    });
    points.rotation.set(burritoRotation.x, burritoRotation.y, burritoRotation.z);
    points.updateMatrix(); 
    points.geometry.applyMatrix( points.matrix );
    points.matrix.identity();
    points.position.set( 0, 0, 0 );
    points.rotation.set( 0, 0, 0 );
    points.scale.set( 1, 1, 1 );
    loadedCallback(points);
  });
}

function setupControls(camera, renderer) {
  let controls = new THREE.TrackballControls( camera, renderer.domElement );
  controls.rotateSpeed = 1.0;
  controls.zoomSpeed = 1.2;
  controls.panSpeed = 0.8;
  controls.noZoom = false;
  controls.noPan = false;
  controls.staticMoving = true;
  controls.dynamicDampingFactor = 0.3;
  controls.keys = [ 65, 83, 68 ];
  return controls;
}

function main() {

  const numPts = 50;
  const ptRadius = 2;
  const minFillRatio = 0.3;
  const lineWidth = 2;
  const ptSplitMultiple = 10;
  const blockDim = new THREE.Vector3(0.2, 0.2, 0.2);
  const moveVec = new THREE.Vector2(0, 5);

  const canvas = document.getElementById('canvas');

  let aspect = canvas.width / canvas.height;
  let camera = new THREE.PerspectiveCamera(45, aspect, 1, 5000);
  let renderer = new THREE.WebGLRenderer({ alpha: true, canvas: canvas });
  let controls = setupControls(camera, renderer);
  let scene = new THREE.Scene();

  function line(vStart, vEnd, color, lineWidth) {
    var material = new THREE.LineBasicMaterial({ color: color, linewidth: lineWidth });
    var geometry = new THREE.Geometry();
    geometry.vertices.push(vStart);
    geometry.vertices.push(vEnd);
    return new THREE.Line(geometry, material);
  }

  loadPoints(points => {
    let occupancyGrid = new OccupancyGrid(points, blockDim);
    occupancyGrid.drawBoundingBox(scene);
    occupancyGrid.drawGrid(scene);
    scene.add(points);
  });

  camera.position.set(0, 2, 10);

  function render() {
    renderer.render(scene, camera);
    controls.update();
    requestAnimationFrame(render);
  }

  render();
}

function rotationInput(element, axis) {
  if (points) {
    switch (axis) {
      case 'x': points.rotation.x = element.value; break;
      case 'y': points.rotation.y = element.value; break;
      case 'z': points.rotation.z = element.value; break;
    }
    points.geometry.verticesNeedUpdate = true;
  }
}

main();
