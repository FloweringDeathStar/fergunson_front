import Head from 'next/head'
import Image from 'next/image'
import styles from '../styles/Home.module.css'
import { useState, useRef, useEffect } from 'react'
import * as tf from '@tensorflow/tfjs'
import * as d3 from 'd3'

export default function Home() {

    const [dataStatus, setDataStatus] = useState('unloaded')
    const [data, setData] = useState([])
    const [julian, setJulian] = useState([]) 


    const handleUpload = async () => {
        let  data_array =[]
        const fileInput = document.createElement('input')
        fileInput.type = 'file'
        fileInput.accept = '.csv'
        fileInput.onchange = async (e) => {
            const file = e.target.files[0]
            const reader = new FileReader()
            reader.readAsText(file)
            reader.onload = async (e) => {
                const text = (e.target.result)
                await setDataStatus('loaded')

                const data = text.split('\r\n').map((row) => {
                    let line =row.split(',')
                    line[0] = line[0].replace(/-/g, '/').split(' ')
                    line[0][0] = line[0][0].split('/')
                    line[0][0] = line[0][0].reverse()
                    line[0][2] = line[0][0][1]*30+line[0][0][2]*1
                    line[0][0] = line[0][0].join('-')
                    line[0][1] = line[0][1].split(':')
                    return line
                })
                //eliminate the first row
                const header_data = data.shift()

                const data_day = data.map((row) => {
                    return row[0][2]
                })
                const data_day_normal = data.map((row) => {
                    return row[0][0]
                })

                
                const data_temp = data.map((row) => {
                    let x = +row[1]
                    x = x.toFixed(1)
                    return +x
                })

                const data_day_t = tf.tensor(data_day,[data_day.length,1], 'int32')
                const data_day_t_unique = tf.unique(data_day_t,0)
                const data_temp_t = tf.tensor(data_temp,[data_temp.length,1], 'float32')
                const  thend = tf.scalar(15.1)
                const theco = tf.scalar(11.2)
                const ka1 = 0.1
                const ka2 = 0.001
                const kd1 = 0.001
                const kd2 = 0.09
                const hmax = -22
                const hmin = -5
                const day_init = tf.scalar(data_day[0])
                const day_end = tf.scalar(data_day[data_day.length-1])

                const edb = tf.scalar(189)
                const data_edb = edb.sub(day_init)
                const data_end = day_end.sub(edb)
                const data_edb_drop = data_edb.dataSync()
                const data_end_drop = data_end.dataSync()


                const data_temp_t_reshape = data_temp_t.reshape([data_temp_t.shape[0]/24,24])
                const data_temp_t_reshape_min = data_temp_t_reshape.min(1)
                const data_temp_t_reshape_max = data_temp_t_reshape.max(1)
                const data_temp_t_reshape_min_max = tf.stack([data_temp_t_reshape_min,data_temp_t_reshape_max],1)
                const data_temp_t_average = data_temp_t_reshape_min_max.mean(1)

                const drop = data_temp_t_average.shape[0]-data_edb_drop[0]-data_end_drop[0]
                const drop_in = data_edb_drop[0]+data_end_drop[0]

                const [data_temp_t_average_reshape, data_temp_t_average_drop] = tf.split(data_temp_t_average,[drop_in,drop],0)
              

                const [data_temp_t_before, data_temp_t_after] = tf.split(data_temp_t_average_reshape,[data_edb_drop[0],data_end_drop[0]],0)

                const data_temp_t_before_threshold = data_temp_t_before.sub(thend)
                //conver data_temp_t_before_threshold to array
                const data_temp_t_before_threshold_array = data_temp_t_before_threshold.dataSync()

                const data_temp_t_after_threshold = data_temp_t_after.sub(theco)
                //conver data_temp_t_after_threshold to array
                const data_temp_t_after_threshold_array = data_temp_t_after_threshold.dataSync()

                let data_LTE =[hmin]

                data_temp_t_before_threshold_array.forEach((element, index) => {
                    let hh = 0
                    let hc = 0

                    if(element<0){
                        let logistic = (hmin-data_LTE[index])/(hmin-hmax)^1.2
                        hh = element*ka1*logistic
                    }else{
                        hh=0
                    }

                    if(element>0){
                        let logistic = (data_LTE[index]-hmax)/(hmin-hmax)
                        hc = element*kd1*logistic
                    }else{
                        hc=0
                    }
                        let lte = data_LTE[index]+hh+hc
                        data_LTE.push(lte)
                })

                data_temp_t_after_threshold_array.forEach((element, index) => {
                    let hh = 0
                    let hc = 0
                    let i = index+data_temp_t_before_threshold_array.length

                    if(element<0){
                        let logistic = (hmin-data_LTE[i])/(hmin-hmax)
                        hh = element*ka2*logistic
                    }else{
                        hh=0
                    }

                    if(element>0){
                        let logistic = (data_LTE[i]-hmax)/(hmin-hmax)^1.2
                        hc = element*kd2*logistic
                    }else{
                        hc=0
                    }

                    let lte = data_LTE[i]+hh+hc
                    data_LTE.push(lte)
                // data_temp_t_average.print()
                })
                const data_temp_average = data_temp_t_average.dataSync()

                const data_day_array =data_day_t_unique.values.dataSync()
                let data_array =[]
                 data_day_array.forEach((element, index) => {
                    let array=[]
                    array.push(element, +data_temp_average[index].toFixed(2), +data_LTE[index].toFixed(2))
                    data_array.push(array)
                })
                setDataStatus("d3")
                setData(data_array)

             
            }
        }
        fileInput.click()
        console.log(data)

    }
    const svgRef = useRef()
    const svgRef2 = useRef()
    useEffect(() => {
        const w =400
        const h = 200
        const svg = d3.select(svgRef.current)
            .attr("width", w)
            .attr("height", h)
            .style("background-color", "#d3d3d3")
            .style("border", "1px solid black")
            .style("margin-top", "20px")
            .style("margin-bottom", "20px")
            .style("overflow", "visible")

        const xScale = d3.scaleLinear()
            .domain([0, data.length-1])
            .range([0, w])

        const yScale = d3.scaleLinear()
            .domain([d3.min(data, d => d[1]), d3.max(data, d => d[1])])
            .range([h, 0])
        const generateScaledLine = d3.line()
            .x((d, i) => xScale(i))
            .y(d => yScale(d[1]))
            .curve(d3.curveMonotoneX)
        
        svg.selectAll(".line")
            .data([data])
            .join("path")
            .attr("d", d=>generateScaledLine(d))
            .attr("fill", "none")
            .attr("stroke", "blue")
        
        const xAxis = d3.axisBottom(xScale)
            .ticks(10)
            .tickFormat(i => data[i][0])
        
        const yAxis = d3.axisLeft(yScale)
            .ticks(10)

        svg.append("g")
            .call(xAxis)
            .attr("transform", `translate(0, ${h})`)

        svg.append("g")
            .call(yAxis)
            .attr("transform", `translate(0, 0)`)

        const svg2 = d3.select(svgRef2.current)
            .attr("width", w)
            .attr("height", h)
            .style("background-color", "#d3d3d3")
            .style("border", "1px solid black")
            .style("margin-top", "20px")
            .style("overflow", "visible")

        const yScale2 = d3.scaleLinear()
            .domain([d3.min(data, d => d[2]), d3.max(data, d => d[2])])
            .range([h, 0])

        const generateScaledLine2 = d3.line()
            .x((d, i) => xScale(i))
            .y(d => yScale2(d[2]))
            .curve(d3.curveMonotoneX)

        svg2.selectAll(".line")
            .data([data])
            .join("path")
            .attr("d", d=>generateScaledLine2(d))
            .attr("fill", "none")
            .attr("stroke", "blue")
        const yAxis2 = d3.axisLeft(yScale)
            .ticks(10)
        svg2.append("g")
            .call(xAxis)
            .attr("transform", `translate(0, ${h})`)
        svg2.append("g")
            .call(yAxis2)
            .attr("transform", `translate(0, 0)`)

    
        console.log(data)
    }, [data])



  
    return (
        <>
        <div className={styles.container}>
            <Head>
                <title>Carculador de Frio Acumulado</title>
                <meta name="description" content="frio acumulado" />
                <link rel="icon" href="/favicon.ico" />
            </Head>
            <h1 className={styles.title}>
                        Carculador de Frio Acumulado
            </h1>
            {dataStatus === 'unloaded' &&
                <main className={styles.main}>

                    <div className={styles.upLoadFileButton}>
                        <button onClick ={handleUpload}>subir archivos</button>
                    </div>
                </main>

            }
            {dataStatus === 'loaded' &&
                <main className={styles.main}>
                    <p>cargado</p>
                </main>
            }
            {dataStatus === 'd3' &&
                <main className={styles.main}>
                    <h2>Temperaturas Promedio</h2>
                    <svg ref={svgRef}>
                    </svg>
                    <h2>LTE</h2>
                    <svg ref ={svgRef2}>
                    </svg>
                </main>
            }
        </div>
        <style jsx>{`
            .upLoadFileButton {

                display: flex;
                justify-content: center;
                align-items: center;
                width: 100%;
                height: 100%;
                background-color: #f0f0f0;
                border-radius: 10px;
                border: 1px solid #ccc;
                cursor: pointer;
            }
            .upLoadFileButton:hover {

                background-color: #e0e0e0;
            }
            .upLoadFileButton:active {
            }
        `}</style>
        </>
    )
}
