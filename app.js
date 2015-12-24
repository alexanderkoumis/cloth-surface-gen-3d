var express = require('express');
var app = express();
var server = require('http').createServer(app);

app.use(express.static(__dirname + '/public'));

server.listen(3000, function() {
    console.log("Started server on port 3000");
});
