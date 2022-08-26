const { count } = require("console");
const express = require("express");
const app = express();
const path = require("path");

//

//
let counter = 0;

app.use(express.static('public'));
app.get("/", (req, res) => {
    res.render("index.ejs", {index: 5, site_count: 0, site_names: [], site_time: [], site_status: []
        ,job_count: 0, job_id: [], job_submit_time: [], job_running_time: [], job_status: []});
    // res.sendFile(path.join(__dirname, "index.html"));
});
app.get("/training_log", (req, res) => {
    res.render("training_log.ejs", 
        {index: 5, site_count: 0, site_names: [], site_time: [], site_status: []});
    // res.sendFile(path.join(__dirname, "index.html"));
});

app.post("/starttrain",(req, res) => {

    var shell = require('shelljs');
    shell.cd('~/workspace/example_project/prod_00/admin@nvidia.com/startup'); //改成相對應的資料夾
    const { stdout, stderr, code } = shell.exec('./fl_admin.sh <test.txt', { silent: true });
    let arr = stdout.split('\n');
    console.log("==========================");

    for(let i = 0; i < arr.length; i++){
        // console.log("==========================");
        console.log(i + " " + arr[i]);
        // console.log("==========================");
    }
    // console.log(arr[0]);
    // console.log(arr[1]);
    console.log("==========================");
    let arr2 = arr[20].split(' ');
    for(let i = 0; i < arr2.length; i++){
        console.log(arr2[i] + '\n');
    }
});

app.post("/submit_job" ,(req, res) => {

    var shell = require('shelljs');
    shell.cd('~/workspace/example_project/prod_00/admin@nvidia.com/startup'); //改成相對應的資料夾
    const { stdout, stderr, code } = shell.exec('./fl_admin.sh <submit_job.txt', { silent: true });
    let arr = stdout.split('\n');
    let arr2 = arr[3].split(' ');
    console.log("==========================");

    for(let i = 0; i < arr2.length; i++){
        // console.log("==========================");
        console.log(i + " " + arr2[i]);
        // console.log("==========================");
    }
    console.log("==========================");
    shell.cd();
    let dt = new Date();
    var fs = require('fs');
    fs.appendFile('log/test.txt', arr2[3] + ' ' + dt + '\n', function(err){
        if(err)
            console.log(err);
        else
            console.log("Done");
    });
} );

app.post("/check_status" ,(req, res) => {

    //
    var shell = require('shelljs');
    shell.cd('~/workspace/example_project/prod_00/admin@nvidia.com/startup'); //改成相對應的資料夾
    const { stdout, stderr, code } = shell.exec('./fl_admin.sh <check_status.txt', { silent: true });
    //
    //console.log(stdout);
    let arr = stdout.split('\n');
    let arr2 = arr[9].split(' ');
    let site_count = Number(arr2[2]);
    let temp = arr[13].split('|');
    console.log(temp);
    let site_names = [];
    let site_time = [];
    let site_status = [];
    //

    let job_count = arr.length - 28 - 2 * site_count;
    let job_id = [];
    let job_submit_time = [];
    let job_running_time = [];
    let job_status = [];
    //

    for(let i = 0; i < site_count; i++){
        let temp = arr[i+13].split('|');
        site_names.push(temp[1]);
        site_time.push(temp[3]);
        temp = arr[i + 18 + site_count].split('|');
        site_status.push(temp[4]);
    }

    for(let i = 0; i < job_count; i++){
        let temp = arr[i + 23 + 2*site_count].split('|');
        job_id.push(temp[1]);
        job_submit_time.push(temp[4]);
        job_status.push(temp[3]);
        job_running_time.push(temp[5]);
    }

    console.log("==========================");
    console.log(job_id);
    console.log(job_submit_time);
    console.log(job_running_time);
    console.log(job_status);
    console.log(site_status);
    // for(let i = 0; i < arr.length; i++){
    //     console.log(i + ' ' + arr[i]);
    // }
    //console.log(arr2);
    console.log("==========================");
    console.log(site_count);


    //


    
    //res.render("training_log.ejs");
    res.render("index.ejs", 
    {index: 5, site_count: site_count, site_names: site_names, site_time: site_time, site_status: site_status
    ,job_count: job_count, job_id: job_id, job_submit_time: job_submit_time, job_running_time: job_running_time, job_status: job_status});
} );

app.post("/check_client" ,(req, res) => {

    var shell = require('shelljs');
    shell.cd('~/workspace/example_project/prod_00/admin@nvidia.com/startup'); //改成相對應的資料夾
    const { stdout, stderr, code } = shell.exec('./fl_admin.sh <check_client.txt', { silent: true });
    let arr = stdout.split('\n');
    console.log("==========================");
    for(let i = 0; i < arr.length; i++){
        console.log(i + " " + arr[i]);
    }
    console.log("==========================");
} );

app.post("/check_server" ,(req, res) => {

    var shell = require('shelljs');
    shell.cd('~/workspace/example_project/prod_00/admin@nvidia.com/startup'); //改成相對應的資料夾
    const { stdout, stderr, code } = shell.exec('./fl_admin.sh <check_server.txt', { silent: true });
    let arr = stdout.split('\n');
    console.log("==========================");
    for(let i = 0; i < arr.length; i++){
        console.log(i + " " + arr[i]);
    }
    console.log("==========================");
} );

// app.post("/check_client" ,(req, res) => {

//     var shell = require('shelljs');
//     shell.cd('~/workspace/example_project/prod_00/admin@nvidia.com/startup'); //改成相對應的資料夾
//     const { stdout, stderr, code } = shell.exec('./fl_admin.sh <check_client.txt', { silent: true });
//     let arr = stdout.split('\n');
//     console.log("==========================");
//     for(let i = 0; i < arr.length; i++){
//         console.log(i + " " + arr[i]);
//     }
//     console.log("==========================");
// } );

app.post("/test",(req, res) => {

    //shell.cd('~/workspace/example_project/prod_00/admin@nvidia.com/startup'); //改成相對應的資料夾
    // counter++;
    // const { stdout, stderr, code } = shell.exec('ls', { silent: true });
    // console.log("==========================");
    // console.log(typeof stdout);
    // console.log("==========================");
    // console.log("hiiii");
    // res.render("index.ejs", {index: counter, site_count: 5, site_names: [], site_time: [], site_status: []});

    //
    var shell = require('shelljs');
    shell.cd('~/workspace/example_project/prod_00/admin@nvidia.com/startup'); //改成相對應的資料夾
    const { stdout, stderr, code } = shell.exec('./fl_admin.sh <list_jobs.txt', { silent: true });

    let arr = stdout.split('\n');
    let arr2 = arr[6].split('|');
    console.log(arr2);

    let job_id = [];
    let job_submit_time = [];
    let job_running_time = [];
    let job_status = [];

    for(let i = 0; i < arr.length - 11; i++){
        let temp = arr[i+6].split('|');
        job_id.push(temp[1]);
        job_submit_time.push(temp[4]);
        job_status.push(temp[3]);
        job_id.push(temp[5]);
    }

    // for(let i = 0; i < arr.length; i++){
    //     console.log(i + arr[i]);
    // }

    //console.log(stdout);


    //
});

app.listen(3000, () => {
    //
    console.log("Server is running 33333333");

    let dt = new Date();
    //console.log(dt_string);
    var fs = require('fs');
    fs.appendFile('log/test.txt', dt + '\n', function(err){
        if(err)
            console.log(err);
        else
            console.log("Done");
    });
});