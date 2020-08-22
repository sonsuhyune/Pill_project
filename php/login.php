<?php

header('content-type: text/html; charset=utf-8'); 

$connect=mysqli_connect( "localhost", "pill", "pilldb", "userinfo") or 
    die( "SQL server에 연결할 수 없습니다.");
 
mysqli_query("SET NAMES UTF8");
 
// 세션 시작
session_start();
 
$id = $_POST[u_id];
$sql = "SELECT userpw FROM person WHERE userid = '$id'";
 
$result = mysqli_query($connect, $sql);
 
// result of sql query
if($result)
{
    $row = mysqli_fetch_array($result);
    if(is_null($row[userpw]))
    {
      echo "Can not find ID";
    }
    else
    {
      echo "$row[userpw]";
    }
}
else
{
   echo mysqli_errno($connect);
}

?>
