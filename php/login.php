<?php
  header('content-type: text/html; charset=utf-8');
  // 데이터베이스 접속 문자열. (db위치, 유저 이름, 비밀번호)
  $connect=mysqli_connect( "localhost", "pill", "pilldb", "userinfo") or die( "SQL server에 연결할 수 없습니다.");

  mysqli_query("SET NAMES UTF8");

  // 세션 시작
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
      echo "$row[pw_chk]";   // 0이면 비밀번호 불일치, 1이면 일치
    }
  }
  else
  {
   echo mysqli_errno($connect);
  }
?>
