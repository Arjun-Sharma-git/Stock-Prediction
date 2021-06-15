const express = require("express"); //requires the express module

var request = require('request'); //used for http request in this case

var fs = require('fs');

const ejs = require('ejs');

const converter = require('json-2-csv'); //passes json to csv

const axios = require('axios'); //requires the axios module which helps in making get and post requests from API

const execa = require('execa');

const https = require("https"); //works similarly like the axios module

const bodyParser = require("body-parser"); //body-parser helps in prsing all the data to the backend which the user enters

const {
  spawn
} = require('child_process');

var app = express();

//without this piece of code boudyparser will not work
app.use(bodyParser.urlencoded({
  extended: true
}));

app.use(express.static("public"));

app.set('view engine', 'ejs');

//runs the server on port 6969
app.listen(process.env.PORT || 6969, function() {
  console.log("Server is Running on port 6969");
});


//this get requests gets the html file from which the user entered data is coming and sends it for backend work
app.get("/", function(req, res) {
  res.sendFile(__dirname + "/index.html");
});

app.get("/result", function(req, res) {
  res.render("result");
})


//this part here handles all the code of getting data from marketstack API
app.post("/", function(req, resp) {

  var symbol = req.body.company; //the company symbol which the user enters
  var url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&datatype=csv&symbol=" + symbol + "&apikey=E3SOUHEOPD38CMZ4"; //api url
  var csv = "";
  //generates request for
  request.get({
    url: url,
    headers: {
      'User-Agent': 'request'
    }
  }, (err, res, info) => {
    if (err) {
      console.log('Error:', err);
    } else if (res.statusCode !== 200) {
      console.log('Status:', res.statusCode);
    } else {
      // data is successfully parsed as a csv object:
      // console.log(data);
      //   const subprocess = execa('python', ['./test.py', data]);
      //   subprocess.stdout.pipe(process.stdout);
      //   (async () => {
      // const {stdout} = await subprocess;

      const python = spawn('python', ['test.py', info]);
      // collect data from script
      python.stdout.on('data', function(data) {
        console.log('Pipe data from python script ...');
        dataToSend = data.toString();
        console.log(dataToSend);
      });
      // in close event we are sure that stream from child process is closed
      python.on('close', (code) => {
        console.log(`child process close all stdio with code ${code}`);
        // send data to browser
      });



      // // Returning Result
      // console.log('child output:', stdout.toString()

      //if json to csv but data is already recieved in csv
      //------------------------------------------
      // convert JSON array to CSV string        |
      // converter.json2csv(data, (err, csv) => {|
      //   if (err) {                            |
      //     throw err;                          |
      //   }                                     |
      //                                         |
      //   // print CSV string                   |
      //   console.log(csv);                     |
      // });                                     |
      //------------------------------------------

    }
  });
  resp.render("result");
});
