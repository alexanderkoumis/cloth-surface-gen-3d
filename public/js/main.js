'use strict'

let pointCloud = null;

function computeCentroid(pointCloud) {
  let centroid = new THREE.Vector3();
  const vertices = pointCloud.geometry.vertices;
  pointCloud.geometry.vertices.forEach(vertice => {
    centroid.add(vertice);
  });
  centroid.divideScalar(vertices.length);
  return centroid;
}

function loadPointCloud(loadedCallback) {
  new THREE.PLYLoader().load('ply/burrito_trimmed.ply', function (geometry) {
    pointCloud = new THREE.Points(geometry, new THREE.PointsMaterial({
      size: 0.1,
      vertexColors: THREE.VertexColors
    }));
    const centroid = computeCentroid(pointCloud);
    pointCloud.geometry.vertices.forEach(vertice => {
      vertice.sub(centroid);
    });
    const burritoRotation = new THREE.Euler(2.1, 0.0584, 2.0584)
    pointCloud.rotation.set(burritoRotation.x, burritoRotation.y, burritoRotation.z);
    pointCloud.updateMatrix(); 
    pointCloud.geometry.applyMatrix(pointCloud.matrix);
    pointCloud.matrix.identity();
    pointCloud.position.set(0, 0, 0);
    pointCloud.rotation.set(0, 0, 0);
    pointCloud.scale.set(1, 1, 1);
    loadedCallback(pointCloud);
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
  const moveScalar = 5;
  const pointSize = 0.1;
  const blockDimGrid = new THREE.Vector3(0.2, 0.2, 0.2);
  const blockDimCloth = new THREE.Vector3(0.1, 0.1, 0.1);

  const canvas = document.getElementById('canvas');

  let aspect = canvas.width / canvas.height;
  let camera = new THREE.PerspectiveCamera(45, aspect, 1, 5000);
  let renderer = new THREE.WebGLRenderer({ alpha: true, canvas: canvas });
  let controls = setupControls(camera, renderer);
  let scene = new THREE.Scene();

  let occupancyGrid = null;
  let cloth = null;

  loadPointCloud(pointCloud => {
    occupancyGrid = new OccupancyGrid(pointCloud, blockDimGrid);
    cloth = new Cloth(blockDimCloth, occupancyGrid.boundingBox,
                      moveScalar, pointSize);

    let points = cloth.genPoints();
    scene.add(points);
    occupancyGrid.drawBoundingBox(scene);
    occupancyGrid.drawGrid(scene);
    scene.add(pointCloud);
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
  if (pointCloud) {
    switch (axis) {
      case 'x': pointCloud.rotation.x = element.value; break;
      case 'y': pointCloud.rotation.y = element.value; break;
      case 'z': pointCloud.rotation.z = element.value; break;
    }
    pointCloud.geometry.verticesNeedUpdate = true;
  }
}

main();
