<?php

header('content-type: text/html; charset=utf-8'); 

$connect=mysqli_connect('localhost', 'pill', 'pilldb', 'userinfo') or  
    die( "SQL server에 연결할 수 없습니다.");

mysqli_query("SET NAMES UTF8");

session_start();

$id = $_POST[u_id];
$pw = $_POST[u_pw];

$sql = "INSERT INTO person(userid, userpw) VALUES('$id', '$pw')";

$result = mysqli_query($connect, $sql);

// result of sql query
echo mysqli_errno($connect);

?>
