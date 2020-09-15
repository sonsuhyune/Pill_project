<?php

header('content-type: text/html; charset=utf-8');

$connect=mysqli_connect('localhost', 'pill', 'pilldb', 'userinfo') or
    die( "SQL server.");

mysqli_query("SET NAMES UTF8");

session_start();

$id = $_POST[u_id];
$nick = $_POST[u_nick];
$pill_name = $_POST[pill_name];
$img_path = $_POST[img_path];


$sql_chk = INSERT INTO pill(user_id,pill_name,pill_nickname, img) VALUES ($id, $pill_name, $nick, $img_path);

#$result_chk = mysqli_query($connect, $sql_chk);
#echo "0";

?>
