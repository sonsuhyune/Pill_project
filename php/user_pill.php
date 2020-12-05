<?php
  
$connect=mysqli_connect( "localhost", "pill", "pilldb", "userinfo");

mysqli_query("SET NAMES UTF8");

session_start();

$id = $_POST[u_id];

$sql = "SELECT pill_name,pill_nickname,img FROM pill WHERE user_id = '$id'";

$result = mysqli_query($connect,$sql);

$data = array();

if($result){

    while($row=mysqli_fetch_array($result)){
        array_push($data,
            array('pill_name'=>$row['pill_name'],
            'pill_nickname'=>$row['pill_nickname'],
            'img'=>$row['img'] ));
    }

    header('Content-Type: application/json; charset=utf8');
    $json = json_encode(array("user_result"=>$data), JSON_PRETTY_PRINT+JSON_UNESCAPED_UNICODE);
    echo $json;

}
else{
    echo "SQL error : ";
    echo mysqli_error($connect);
}
?>