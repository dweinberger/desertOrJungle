

// Drag and drop code from:
// https://www.smashingmagazine.com/2018/01/drag-drop-file-uploader-vanilla-js/

var dropArea;
var img = new Image(); 
var gModelPath = './models/desertVjungle01.json';


// ************************ Drag and drop ***************** //
function init(){

	dropArea = document.getElementById("dropzone_element");


	// Prevent default drag behaviors
	['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
	  dropArea.addEventListener(eventName, preventDefaults, false)   
	  document.body.addEventListener(eventName, preventDefaults, false)
	})

	// Highlight drop area when item is dragged over it
	;['dragenter', 'dragover'].forEach(eventName => {
	  dropArea.addEventListener(eventName, highlight, false)
	})

	;['dragleave', 'drop'].forEach(eventName => {
	  dropArea.addEventListener(eventName, unhighlight, false)
	})

	// Handle dropped files
	dropArea.addEventListener('drop', handleDrop, false)}
	
	loadModel();

function preventDefaults (e) {
  e.preventDefault()
  e.stopPropagation()
}

function highlight(e) {
  dropArea.classList.add('highlight')
}

function unhighlight(e) {
  dropArea.classList.remove('active')
}

function handleDrop(e) {
	var dt = e.dataTransfer
	var files = dt.files
	var imgfile = files[0];
	$(imgfile).css({"object-fit" : "contain"});
	$("#imagefilename").text(imgfile.name);
  	handleFiles(imgfile);
}


function handleFiles(imgfile) {
 	let imgsrc=previewFile(imgfile);

}

async function previewFile(file) {
	$("#dropzone_element").html("");
  let reader = new FileReader()
  reader.readAsDataURL(file)
  reader.onload = function() {
  		let img = document.createElement('img');
		img.src = reader.result
		document.getElementById('dropzone_element').appendChild(img)
		//document.getElementById('gallery50').appendChild(img)
		console.log("preview source: " + img.src.substr(0,15));
		createTensor(img);
  }
  		
}


var themodel;

async function loadModel(){

	themodel = await tf.loadModel(gModelPath);
	$("#modelname").text(gModelPath);
    console.log("model uploaded:  " + themodel);

}

function createTensor(newImage){
	//console.log("tensor img source: " + newImage.src.substr(0,15));
	var canvas = document.createElement('canvas');
	canvas.height = 50;
	canvas.width = 50;
	var ctx = canvas.getContext('2d');
	var pixelData = [];

	ctx.drawImage(newImage, 0, 0,50,50);
	var imageData = ctx.getImageData(0,0,50,50);
	var uint8data = imageData.data;
	for (var j=0; j < uint8data.length; j+=4){
		pixelData.push(uint8data[j] / 255);
		pixelData.push(uint8data[j+1] / 255);
		pixelData.push(uint8data[j+2] /255);
		}
	
	console.log('done with imagedata', pixelData.length)
	var allImagesTensor = tf.tensor(pixelData, [1,50,50,3]);
	
	themodel.predict(allImagesTensor)
	const res = themodel.predict(allImagesTensor)
	const answerArray = res.dataSync()
	console.log(answerArray)
	
	displayResults(answerArray)

	console.log(answerArray[0] + " : " + answerArray[1]);
	

}

function displayResults(answers){
	var r0 = answers[0].toFixed(3);
	var r1 = answers[1].toFixed(3) ;

	$("#personCtr").text(r0);
	$("#animalCtr").text(r1);

	
	// bar chart
	$(".bar").fadeIn();
	$("#persline").animate({width : Math.round(answers[0] * 600)},500);
	$("#aniline").animate({width : Math.round(answers[1] * 600)},750);

}
