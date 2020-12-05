<?php
  header('content-type: text/html; charset=utf-8');
  //  QM 문P. (dbD,  @ , )
  $connect=mysqli_connect( "localhost", "pill", "pilldb", "userinfo") or die( "SQL server[34m~W~P [34m~W[34m~U|  [34m~H~X [34m~W~F[34m~Jm~K~H[34m~K5;1H.");

  mysqli_query("SET NAMES UTF8");

  // X \Q
  session_start();

  $id = $_POST['u_id'];
  $pw = $_POST['u_pw'];

  $sql = "SELECT IF(strcmp(userpw,'$pw'),0,1) pw_chk FROM person  WHERE userid = '$id'";

  $result = mysqli_query($connect, $sql);

  // 쿼리 결과
  if($result)
  {
    $row = mysqli_fetch_array($result);
    if(is_null($row[pw_chk]))
    {
      echo "Can not find ID";
    }
    else
    {
      echo "$row[pw_chk]";   // 0면  , 1면 [m
    }
  }
  else
  {
   echo mysqli_errno($connect);
  }
?>