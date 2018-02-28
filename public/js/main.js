'use strict'

let cloth = null;
let scene = new THREE.Scene();

let sceneObjs = {
  'pts': { show: true, obj: null },
  'mesh': { show: true, obj: null },
  'bbox': { show: true, obj: null },
  'verts': { show: true, obj: null },
  // 'grid': { show: true, obj: null }
}

function toggle(objName) {
  let obj = sceneObjs[objName];
  obj['show'] = obj['show'] ? false : true;
  if (obj['obj']) {
    if (obj['show']) {
      scene.add(obj['obj']);
    }
    else {
      scene.remove(obj['obj']);
    }
  }
}

function rotationInput(element, axis) {
  let pointCloud = sceneObjs['pts']['obj'];
  if (pointCloud) {
    switch (axis) {
      case 'x': pointCloud.rotation.x = element.value; break;
      case 'y': pointCloud.rotation.y = element.value; break;
      case 'z': pointCloud.rotation.z = element.value; break;
    }
    pointCloud.geometry.verticesNeedUpdate = true;
  }
}


function loadPointCloud(loadedCallback) {

  function computeCentroid(points) {
    let centroid = new THREE.Vector3();
    for (let i = 0; i < points.length; i += 3) {
      centroid.x += points[i];
      centroid.y += points[i+1];
      centroid.z += points[i+2];
    }
    centroid.divideScalar(points.length / 3);
    return centroid;
  }

  new THREE.PLYLoader().load('ply/burrito_trimmed.ply', geometry => {
    let pointCloud = new THREE.Points(geometry, new THREE.PointsMaterial({
      size: 0.04,
      vertexColors: THREE.VertexColors
    }));

    const points = pointCloud.geometry.attributes.position.array;
    const centroid = computeCentroid(points);

    for (let i = 0; i < points.length; i += 3) {
      points[i] -= centroid.x;
      points[i+1] -= centroid.y;
      points[i+2] -= centroid.z;
    }

    const burritoRot = new THREE.Euler(2.1 + Math.PI / 2, 0.0584, 2.0584)

    pointCloud.rotation.set(burritoRot.x, burritoRot.y, burritoRot.z);
    pointCloud.updateMatrix(); 
    pointCloud.geometry.applyMatrix(pointCloud.matrix);
    pointCloud.matrix.identity();
    pointCloud.position.set(0, 0, 0);
    pointCloud.rotation.set(0, 0, 0);
    pointCloud.scale.set(1, 1, 1);
    loadedCallback(pointCloud);
  });
}

function main() {

  let canvas = document.getElementById('canvas');
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;

  let aspect = canvas.width / canvas.height;
  let camera = new THREE.PerspectiveCamera(45, aspect, 1, 5000);
  let renderer = new THREE.WebGLRenderer({ alpha: true, canvas: canvas, antialias: true });
  let controls = new THREE.TrackballControls( camera, renderer.domElement );

  renderer.setPixelRatio( window.devicePixelRatio );
  renderer.setSize(window.innerWidth, window.innerHeight);

  camera.position.set(0, 2, 10);
  controls.rotateSpeed = 1.0;
  controls.zoomSpeed = 1.2;
  controls.panSpeed = 0.8;
  controls.noZoom = false;
  controls.noPan = false;
  controls.staticMoving = true;
  controls.dynamicDampingFactor = 0.3;
  controls.keys = [65, 83, 68];

  loadPointCloud(pointCloud => {
    cloth = new Cloth(pointCloud, {
      bboxLineWidth: 2,
      minFillRatio: 0.1,
      maxNeighborDist: 0.05,
      pointSize: 0.01,
      moveVec: new THREE.Vector3(0, 0, -0.01),
      blockDimGrid: new THREE.Vector3(0.03, 0.03, 0.01),
      blockDimCloth: new THREE.Vector3(0.06, 0.06, 0.03)
    });
    sceneObjs['pts']['obj'] = pointCloud;
    sceneObjs['bbox']['obj'] = cloth.occupancyGrid.bboxObj;
    sceneObjs['mesh']['obj'] = cloth.mesh;
    sceneObjs['verts']['obj'] = cloth.points;
    // sceneObjs['grid']['obj'] = new THREE.GridHelper(10, 10);
    scene.add(sceneObjs['pts']['obj']);
    scene.add(sceneObjs['bbox']['obj']);
    scene.add(sceneObjs['mesh']['obj']);
    scene.add(sceneObjs['verts']['obj']);
    // scene.add(sceneObjs['grid']['obj']);
    // cloth.occupancyGrid.drawGrid(scene);
  });


  (function render() {
    if (cloth) {
      cloth.update();
    }
    renderer.render(scene, camera);
    controls.update();
    requestAnimationFrame(render);
  })();

}

main();
