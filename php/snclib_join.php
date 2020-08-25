<?php

header('content-type: text/html; charset=utf-8');

$connect=mysqli_connect('localhost', 'pill', 'pilldb', 'userinfo') or
    die( "SQL server에 연결할 수 없습니다.");

mysqli_query("SET NAMES UTF8");

session_start();

$id = $_POST[u_id];
$pw = $_POST[u_pw];

// 기존에 등록된 아이디인지 확인 NULL : 등록되지 않은 아이디
$sql_chk = "SELECT id FROM person WHERE userid = '$id'";

$result_chk = mysqli_query($connect, $sql_chk);

$row = mysqli_fetch_array($result_chk);

if (is_null($row[id]))
{
    $sql = "INSERT INTO person(userid, userpw) VALUES('$id', '$pw')";
    $result = mysqli_query($connect, $sql);
    echo "0";
}
else
{
    echo "$row[id]";
}
?>
