const express = require("express");
const app = express();
const path = require("path");
//

//


app.get("/", (req, res) => {
    res.sendFile(path.join(__dirname, "index.html"));
});

app.post("/starttrain",(req, res) => {
    //res.sendFile(path.join(__dirname, "index.html"));
    //res.send("Tanq");
    // var shell = require('shelljs')

//检查控制台是否以运行`git `开头的命令
    // if (!shell.which('git')) {
    // //在控制台输出内容
    // shell.echo('Sorry, this script requires git');
    // shell.exit(1);
    // }

    // shell.rm('-rf','out/Release');//强制递归删除`out/Release目录`
    // shell.cp('-R','stuff/','out/Release');//将`stuff/`中所有内容拷贝至`out/Release`目录

    // shell.cd('lib');//进入`lib`目录
    // //找出所有的扩展名为js的文件，并遍历进行操作
    // shell.ls('*.js').forEach(function (file) {
    // /* 这是第一个难点：sed流编辑器,建议专题学习，-i表示直接作用源文件 */
    // //将build_version字段替换为'v0.1.2'
    // shell.sed('-i', 'BUILD_VERSION', 'v0.1.2', file);
    // //将包含`REMOVE_THIS_LINE`字符串的行删除
    // shell.sed('-i', /^.*REMOVE_THIS_LINE.*$/, '', file);
    // //将包含`REPLACE_LINE_WITH_MACRO`字符串的行替换为`macro.js`中的内容
    // shell.sed('-i', /.*REPLACE_LINE_WITH_MACRO.*\n/, shell.cat('macro.js'), file);
    // });

    //返回上一级目录

    var shell = require('shelljs');
    //shell.exec('source ~/nvflare-env/bin/activate');
    shell.cd('~/workspace/example_project/prod_00/admin@nvidia.com/startup'); //改成相對應的資料夾
    //shell.exec('./fl_admin.sh <test.txt');
    const { stdout, stderr, code } = shell.exec('./fl_admin.sh <test.txt', { silent: true })
    console.log("--------------------------");
    console.log(stdout);
    console.log("--------------------------");
    // ls.stdin.write('hsad\n');
    // // ls.stdin.write('help\n');
    // // ls.stdin.write('asdsadasd\n');

    // ls.stdout.on('data', function (data) {
    //     //ls.stdin.write('hsad\n');
    //     console.log(data.toString());
// });
    
    //shell.exec('show');

//run external tool synchronously
//即同步运行外部工具
// if (shell.exec('git commit -am "Auto-commit"').code !== 0){
//     shell.echo('Error: Git commit failed');
//     shell.exit(1);
// }

// spawn("ls -ls", (error, stdout, stderr) => {
//     if (error) {
//         console.log(`error: ${error.message}`);
//         return;
//     }
//     if (stderr) {
//         console.log(`stderr: ${stderr}`);
        
//         return;
//     }
//     console.log(`stdout: ${stdout}`);
//     //res.send(`stdout: ${stdout}`);
// });
// //
// exec("source ~/nvflare-env/bin/activate", (error, stdout, stderr) => {
//     if (error) {
//         console.log(`error: ${error.message}`);
//         return;
//     }
//     if (stderr) {
//         console.log(`stderr: ${stderr}`);
        
//         return;
//     }
//     console.log(`stdout: ${stdout}`);
//     //res.send(`stdout: ${stdout}`);
// });
// exec("cd ~/NVFlare/examples/cifar10;./start_fl_secure.sh 8", (error, stdout, stderr) => {
//     if (error) {
//         console.log(`error: ${error.message}`);
//         return;
//     }
//     if (stderr) {
//         console.log(`stderr: ${stderr}`);
        
//         return;
//     }
//     console.log(`stdout: ${stdout}`);
//     //res.send(`stdout: ${stdout}`);
// });

// exec("ls -l", (error, stdout, stderr) => {
//     if (error) {
//         console.log(`error: ${error.message}`);
//         return;
//     }
//     if (stderr) {
//         console.log(`stderr: ${stderr}`);
        
//         return;
//     }
//     console.log(`stdout: ${stdout}`);
//     //res.send(`stdout: ${stdout}`);
// });
});

app.listen(3000, () => {
    //
    console.log("Server is running");
});