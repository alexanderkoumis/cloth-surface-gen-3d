'use strict'

let scene = new THREE.Scene();

let sceneObjs = {
  'pts': { show: true, obj: null },
  'mesh': { show: true, obj: null },
  'bbox': { show: true, obj: null }
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

  function computeCentroid(pointCloud) {
    let centroid = new THREE.Vector3();
    const vertices = pointCloud.geometry.vertices;
    pointCloud.geometry.vertices.forEach(vertice => {
      centroid.add(vertice);
    });
    centroid.divideScalar(vertices.length);
    return centroid;
  }

  new THREE.PLYLoader().load('ply/burrito_trimmed.ply', function (geometry) {
    let pointCloud = new THREE.Points(geometry, new THREE.PointsMaterial({
      size: 0.04,
      vertexColors: THREE.VertexColors
    }));
    const centroid = computeCentroid(pointCloud);
    pointCloud.geometry.vertices.forEach(vertice => {
      vertice.sub(centroid);
    });
    const burritoRot = new THREE.Euler(2.1, 0.0584, 2.0584)
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

  const canvas = document.getElementById('canvas');

  let cloth = null;
  let aspect = canvas.width / canvas.height;
  let camera = new THREE.PerspectiveCamera(45, aspect, 1, 5000);
  let renderer = new THREE.WebGLRenderer({ alpha: true, canvas: canvas });
  let controls = new THREE.TrackballControls( camera, renderer.domElement );

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
      minFillRatio: 0.2,
      maxNeighborDist: 0.1,
      moveScalar: 0.01,
      pointSize: 0.03,
      blockDimGrid: new THREE.Vector3(0.03, 0.03, 0.03),
      blockDimCloth: new THREE.Vector3(0.04, 0.04, 0.03)
    });
    sceneObjs['pts']['obj'] = pointCloud;
    sceneObjs['bbox']['obj'] = cloth.occupancyGrid.bboxObj;
    sceneObjs['mesh']['obj'] = cloth.mesh;
    scene.add(sceneObjs['pts']['obj']);
    scene.add(sceneObjs['bbox']['obj']);
    scene.add(sceneObjs['mesh']['obj']);
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
