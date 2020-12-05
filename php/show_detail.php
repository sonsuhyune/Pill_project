<?php
  
$connect=mysqli_connect( "localhost", "pill", "pilldb", "pill_info");

mysqli_query("SET NAMES UTF8");

session_start();

$id = $_POST[pill];

$sql = "SELECT name,ingredient,efficiency,capacity,company,warning FROM pill_information WHERE name = '$id'";

$result = mysqli_query($connect,$sql);

$data = array();

if($result){

    while($row=mysqli_fetch_array($result)){
        array_push($data,
            array('name'=>$row['name'], 'ingredient'=>$row['ingredient'],'efficiency'=>$row['efficiency'],'capacity'=>$row['capacity'], 'company'=>$row['company'],'warning'=>$row['warning'] ));
    }

    header('Content-Type: application/json; charset=utf8');
    $json = json_encode(array("detail_info"=>$data), JSON_PRETTY_PRINT+JSON_UNESCAPED_UNICODE);
    echo $json;

}
else{
    echo "SQL error : ";
    echo mysqli_error($connect);
}
?>